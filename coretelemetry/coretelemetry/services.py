from datetime import datetime
from pathlib import Path

import yaml
from lib_sql.connection import TryConnectContextManager
from lib_sql.engine import get_sql_engine
from lib_sql.inspection import column_exists, table_exists
from pydantic import BaseModel
from pydantic.dataclasses import dataclass as pydantic_dataclass
from sqlalchemy import text

MAX_RESULT_ROWS = 5000


class DataPoint(BaseModel):
    timestamp: datetime | None
    value: float


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
        not_null: bool = True,
    ):
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()

        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()

        if not table_exists(self.engine, table_name, schema=self.db_cfg.schema):
            raise ValueError(f"Table {table_name} not found in DB")

        if not column_exists(self.engine, table_name, column_name, schema=self.db_cfg.schema):
            raise ValueError(f"Column {column_name} not found in {table_name} not found in DB")

        # Build SELECT clause
        columns = ["time", column_name] if time_col else [column_name]
        select_clause = f"SELECT {', '.join(columns)} FROM {table_name}"

        # Build WHERE clause
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

        # Build ORDER BY and LIMIT
        order_clause = " ORDER BY time DESC"
        limit_clause = " LIMIT 1" if start_time is None and end_time is None else ""

        # Combine all parts
        query = f"{select_clause}{where_clause}{order_clause}{limit_clause};"

        # Prepare and execute
        select_query = text(query).bindparams(**params) if params else text(query)

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

    # TODO: Needs an endpoint
    def refresh(self):
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
            raise FileNotFoundError(f"Configuration file not found: {yaml_file_path}")

        with open(yaml_file_path) as f:
            config_data = yaml.safe_load(f)

        table_name = (config_data or {}).get('metrics', {}).get('table_name')

        if not table_name:
            raise KeyError(f"'metrics.table_name' not found in {yaml_file_path}")

        self.metrics_table_cache[agent_id] = table_name

        return table_name

    async def get_telemetry_data(
        self,
        agent_id: str,
        metric: str,
        start_time: str | None,
        end_time: str | None,
    ):

        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)

        table_name = self._get_metrics_table_name(agent_id)
        raw_data = self.sql_reader.read_single_column(table_name, metric, start_time, end_time)

        return [{"timestamp": row[0], "value": float(row[1])} for row in raw_data]


    async def get_available_metrics(self, agent_id: str) -> dict:
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)
        # Placeholder implementation
        return {"agent_id": agent_id, "data": []}


# Create singleton instance
telemetry_manager = TelemetryManager()


def get_telemetry_manager() -> TelemetryManager:
    return telemetry_manager
