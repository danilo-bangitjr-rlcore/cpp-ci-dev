"""SQL utilities for database operations."""

from lib_sql.connection import TryConnectContextManager
from lib_sql.engine import get_sql_engine
from lib_sql.inspection import column_exists, get_all_columns, table_exists
from pydantic.dataclasses import dataclass as pydantic_dataclass
from sqlalchemy import text


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
    def __init__(self, db_cfg: DBConfig):
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
