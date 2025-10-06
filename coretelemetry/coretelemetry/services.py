from datetime import datetime
from pathlib import Path

import yaml
from lib_sql.connection import TryConnectContextManager
from lib_sql.engine import get_sql_engine
from lib_sql.inspection import column_exists, table_exists
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
        self.db_cfg = DBConfig
        self.engine = get_sql_engine(db_data=db_cfg, db_name=db_cfg.db_name)


    def read_single_column(
        self,
        table_name: str,
        column_name: str,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        time_col: bool = True,
    ):
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()

        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()

        if not table_exists(self.engine, table_name, schema=self.db_cfg.schema):
            raise ValueError(f"Table {table_name} not found in DB")

        if not column_exists(self.engine, table_name, column_name, schema=self.db_cfg.schema):
            raise ValueError(f"Column {column_name} not found in {table_name} not found in DB")

        base_query = f"""
        SELECT
            {"time, " if time_col else ""}
            :column_name
        FROM :table_name
        """

        params = {"column_name": column_name, "table_name": table_name}

        if start_time is None and end_time is None:
            base_query += "ORDER BY time DESC LIMIT 1;"
        else:
            base_query += "WHERE "
            conditions = []

            if start_time is not None:
                conditions.append("time >= :start_time::timestamptz")
                params["start_time"] = start_time

            if end_time is not None:
                conditions.append("time <= :end_time::timestamptz")
                params["end_time"] = end_time

            base_query += " AND ".join(conditions)
            base_query += " ORDER BY time DESC;"

        select_query = text(base_query).bindparams(**params)

        with TryConnectContextManager(self.engine) as connection:
            result = connection.execute(select_query).fetchall()

        if not result:
            raise ValueError(
                f"No data found in table '{table_name}' for column '{column_name}' for time range"
                f"from {start_time or ""} to {end_time or ""} ",
            )

        if len(result) > MAX_RESULT_ROWS:
            raise ValueError(
                f"Result exceeded maximum length of {MAX_RESULT_ROWS} rows",
            )

        return result


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

    # TODO: Needs an endpoint
    def refresh(self):
        self.sql_reader = None
        self.metrics_table_cache = {}

    # Private helper methods
    def _get_metrics_table_name(self, agent_id: str) -> str:
        if agent_id in self.metrics_table_cache:
            return self.metrics_table_cache[agent_id]

        yaml_file_path = self.config_path / f"{agent_id}.yaml"

        if not yaml_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_file_path}")

        with open(yaml_file_path) as f:
            config_data = yaml.safe_load(f)

        table_name = (config_data or {}).get('metrics', {}).get('table_name')

        if not table_name:
            raise KeyError(f"'metrics.table_name' not found in {yaml_file_path}")

        self.metrics_table_cache[agent_id] = table_name

        return table_name

    # Telemetry data methods (to be implemented)
    async def get_telemetry_data(
        self,
        agent_id: str,
        metric: str,
        start_time: str | None,
        end_time: str | None,
    ) -> dict:

        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)

        table_name = self._get_metrics_table_name(agent_id)
        data = self.sql_reader.read_single_column(table_name, metric, start_time, end_time)

        return {
            "agent_id": agent_id,
            "metric": metric,
            "start_time": start_time,
            "end_time": end_time,
            "data": data,
        }

    async def get_available_metrics(self, agent_id: str) -> dict:
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)
        # Placeholder implementation
        return {"agent_id": agent_id, "data": []}


# Create singleton instance
telemetry_manager = TelemetryManager()


def get_telemetry_manager() -> TelemetryManager:
    return telemetry_manager
