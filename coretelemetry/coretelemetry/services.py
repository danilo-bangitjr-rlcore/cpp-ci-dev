from datetime import datetime
from pathlib import Path

import yaml
from fastapi import HTTPException
from lib_sql.connection import TryConnectContextManager
from lib_sql.engine import get_sql_engine
from lib_sql.inspection import column_exists, get_all_columns, table_exists
from pydantic.dataclasses import dataclass as pydantic_dataclass
from sqlalchemy import text

MAX_RESULT_ROWS = 5000


@pydantic_dataclass
class DBConfig:
    drivername: str = 'postgresql+psycopg2'
    username: str = 'postgres'
    password: str = 'password'
    ip: str = 'localhost'
    port: int = 5432
    db_name: str = 'postgres'
    schema: str = 'public'

class SqlReader:
    def __init__(self, db_cfg : DBConfig):
        self.db_cfg = db_cfg
        self.engine = get_sql_engine(db_data=db_cfg, db_name=db_cfg.db_name)

    def table_exists(self, table_name: str) -> bool:
        return table_exists(self.engine, table_name, schema=self.db_cfg.schema)

    def column_exists(self, table_name: str, column_name: str) -> bool:
        return column_exists(self.engine, table_name, column_name, schema=self.db_cfg.schema)

    def build_query(
        self,
        table_name: str,
        column_name: str,
        start_time: str | None,
        end_time: str | None,
        time_col: bool,
        not_null: bool,
    ) -> tuple[str, dict]:
        columns = ["time", column_name] if time_col else [column_name]
        select_clause = f"SELECT {', '.join(columns)} FROM {table_name}"

        where_conditions = []
        params = {}

        if not_null:
            where_conditions.append(f"{column_name} IS NOT NULL")

        if start_time is not None:
            where_conditions.append("time >= :start_time")
            params["start_time"] = start_time

        if end_time is not None:
            where_conditions.append("time <= :end_time")
            params["end_time"] = end_time

        where_clause = f" WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        order_clause = " ORDER BY time DESC"
        limit_clause = " LIMIT 1" if start_time is None and end_time is None else ""

        query = f"{select_clause}{where_clause}{order_clause}{limit_clause};"
        return query, params

    def execute_query(self, query: str, params: dict):
        select_query = text(query).bindparams(**params) if params else text(query)
        with TryConnectContextManager(self.engine) as connection:
            return connection.execute(select_query).fetchall()

    def get_column_names(self, table_name: str) -> list[str]:
        columns = get_all_columns(self.engine, table_name, schema=self.db_cfg.schema)
        return [col["name"] for col in columns]

    def test_connection(self) -> bool:
        try:
            with TryConnectContextManager(self.engine, max_tries=1, backoff_seconds=0) as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


class TelemetryManager:

    def __init__(self):
        self.db_config = DBConfig()
        self.config_path = Path("clean/")
        self.metrics_table_cache: dict[str, str] = {}
        self.sql_reader: SqlReader | None = None

    # Configuration methods
    def get_db_config(self) -> DBConfig:
        return self.db_config

    def set_db_config(self, config: DBConfig) -> DBConfig:
        self.db_config = config
        return self.db_config

    def get_config_path(self) -> Path:
        return self.config_path

    def set_config_path(self, path: Path) -> Path:
        self.config_path = path
        return self.config_path

    def clear_cache(self):
        """Clear all cached data including SQL reader and metrics table cache."""
        self.sql_reader = None
        self.metrics_table_cache = {}

    def test_db_connection(self) -> bool:
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)
        return self.sql_reader.test_connection()

    # Private helper methods
    def _get_metrics_table_name(self, agent_id: str) -> str:
        if agent_id in self.metrics_table_cache:
            return self.metrics_table_cache[agent_id]

        yaml_file_path = self.config_path / f"{agent_id}.yaml"

        if not yaml_file_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Configuration file not found for agent '{agent_id}': {yaml_file_path}",
            )

        try:
            with open(yaml_file_path) as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse configuration file for agent '{agent_id}': {e!s}",
            ) from e

        table_name = (config_data or {}).get('metrics', {}).get('table_name')

        if not table_name:
            raise HTTPException(
                status_code=500,
                detail=f"'metrics.table_name' not found in configuration for agent '{agent_id}'",
            )

        self.metrics_table_cache[agent_id] = table_name

        return table_name

    def get_telemetry_data(
        self,
        agent_id: str,
        metric: str,
        start_time: str | None,
        end_time: str | None,
    ):
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)

        if metric.lower() == "time":
            raise HTTPException(status_code=400, detail="'time' is a reserved column and cannot be used as a metric")

        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()

        table_name = self._get_metrics_table_name(agent_id)

        if not self.sql_reader.table_exists(table_name):
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found in database")

        if not self.sql_reader.column_exists(table_name, metric):
            raise HTTPException(status_code=404, detail=f"Column '{metric}' not found in table '{table_name}'")

        query, params = self.sql_reader.build_query(
            table_name, metric, start_time, end_time, time_col=True, not_null=True,
        )

        try:
            raw_data = self.sql_reader.execute_query(query, params)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Database connection failed: {e!s}") from e

        if not raw_data:
            time_range = f" from {start_time} to {end_time}" if start_time or end_time else ""
            raise HTTPException(
                status_code=404,
                detail=f"No data found in table '{table_name}' for column '{metric}'{time_range}",
            )

        if len(raw_data) > MAX_RESULT_ROWS:
            raise HTTPException(status_code=413, detail=f"Result exceeded maximum length of {MAX_RESULT_ROWS} rows")

        return [{"timestamp": row[0], "value": float(row[1])} for row in raw_data]

    def get_available_metrics(self, agent_id: str) -> dict:
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)

        table_name = self._get_metrics_table_name(agent_id)

        if not self.sql_reader.table_exists(table_name):
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found in database")

        try:
            all_columns = self.sql_reader.get_column_names(table_name)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to retrieve columns: {e!s}") from e

        metrics = [col for col in all_columns if col.lower() != "time"]

        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics available for agent '{agent_id}'")

        return {"agent_id": agent_id, "data": metrics}


# Create singleton instance
telemetry_manager = TelemetryManager()


def get_telemetry_manager() -> TelemetryManager:
    return telemetry_manager
