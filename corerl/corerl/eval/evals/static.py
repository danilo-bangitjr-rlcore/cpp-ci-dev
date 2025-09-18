import json
import logging
from datetime import UTC, datetime

import pandas as pd
from lib_sql.connection import TryConnectContextManager
from lib_sql.engine import get_sql_engine
from lib_sql.utils import SQLColumn, create_tsdb_table_query
from lib_sql.writers.buffered_sql_writer import BufferedSqlWriter
from lib_sql.writers.static_schema_sql_writer import StaticSchemaSqlWriter
from lib_utils.errors import fail_gracefully
from lib_utils.time import now_iso
from sqlalchemy import text

from corerl.eval.evals.base import EvalDBConfig

log = logging.getLogger(__name__)


class StaticEvalsTable:
    """Evals table implementation using StaticSchemaSqlWriter with fixed schema."""

    # ============================================================================
    # Initialization & Configuration
    # ============================================================================

    def __init__(self, cfg: EvalDBConfig):
        self.cfg = cfg

        self.engine = get_sql_engine(db_data=cfg, db_name=cfg.db_name)
        log.info(f"Created engine for database {cfg.db_name}")

        def table_factory(schema: str, table: str, columns: list[SQLColumn]):
            return create_tsdb_table_query(
                schema=schema,
                table=table,
                columns=columns,
                partition_column="evaluator",
                index_columns=["evaluator"],
            )

        initial_columns = [
            SQLColumn(name="time", type="TIMESTAMP WITH TIME ZONE", nullable=False),
            SQLColumn(name="agent_step", type="INTEGER", nullable=False),
            SQLColumn(name="evaluator", type="TEXT", nullable=False),
            SQLColumn(name="value", type="jsonb", nullable=False),
        ]

        static_writer = StaticSchemaSqlWriter(
            engine=self.engine,
            table_name=cfg.table_name,
            columns=initial_columns,
            table_creation_factory=table_factory,
            schema=cfg.table_schema,
        )

        self._writer = BufferedSqlWriter(
            inner=static_writer,
            low_watermark=cfg.low_watermark,
            high_watermark=cfg.high_watermark,
            enabled=True,
        )

    @fail_gracefully()
    def close(self):
        self._writer.close()
        # Dispose of the engine to close all connections before dropping the database
        # This is required in SQLAlchemy 2.0 to prevent "database is being accessed by other users" errors
        self.engine.dispose()

    # ============================================================================
    # Public Write API
    # ============================================================================

    def write(self, agent_step: int, evaluator: str, value: object, timestamp: str | None = None):
        timestamp = timestamp or now_iso()

        # Convert value to JSON string if not already a string
        json_value = value if isinstance(value, str) else json.dumps(value)

        return self._writer.write(
            {
                "time": datetime.fromisoformat(timestamp),
                "agent_step": agent_step,
                "evaluator": evaluator,
                "value": json_value,
            },
        )

    def flush(self) -> None:
        self._writer.flush()

    # ============================================================================
    # Public Read API
    # ============================================================================

    def _execute_read(self, stmt: str) -> pd.DataFrame:
        assert self.engine is not None
        with TryConnectContextManager(self.engine) as connection:
            return pd.read_sql(sql=text(stmt), con=connection)

    def _read_by_eval(self, evaluator: str) -> pd.DataFrame:
        stmt = f"""
            SELECT
                time,
                agent_step,
                value
            FROM {self.cfg.table_name}
            WHERE
                evaluator='{evaluator}';
        """

        df = self._execute_read(stmt)
        df["time"] = pd.to_datetime(df["time"])
        df["agent_step"] = df["agent_step"].astype(int)

        return df

    def _read_by_step(
        self,
        evaluator: str,
        step_start: int | None,
        step_end: int | None,
    ) -> pd.DataFrame:
        stmt = f"""
            SELECT
                agent_step,
                value
            FROM {self.cfg.table_name}
            WHERE
                evaluator='{evaluator}'
        """

        if step_start is not None:
            stmt += f" AND agent_step>='{step_start}'"

        if step_end is not None:
            stmt += f" AND agent_step<='{step_end}'"

        stmt += ";"

        df = self._execute_read(stmt)
        df["agent_step"] = df["agent_step"].astype(int)

        return df

    def _read_by_time(
        self,
        evaluator: str,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> pd.DataFrame:
        stmt = f"""
            SELECT
                time,
                value
            FROM {self.cfg.table_name}
            WHERE
                evaluator='{evaluator}'
        """

        if start_time is not None:
            if start_time.tzinfo is None:
                start_tz = start_time.astimezone().tzinfo
                log.warning(f"naive start_time passed, assuming {start_tz} and converting to UTC")
                start_time = start_time.replace(tzinfo=start_tz).astimezone(UTC)
            stmt += f" AND time>='{start_time.isoformat()}'"

        if end_time is not None:
            if end_time.tzinfo is None:
                end_tz = end_time.astimezone().tzinfo
                log.warning(f"naive end_time passed, assuming {end_tz} and converting to UTC")
                end_time = end_time.replace(tzinfo=end_tz).astimezone(UTC)
            stmt += f" AND time<='{end_time.isoformat()}'"

        stmt += ";"

        df = self._execute_read(stmt)
        df["time"] = pd.to_datetime(df["time"])

        return df

    def read(
        self,
        evaluator: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        # Make sure all points in buffer have been written to DB
        self.flush()

        if start_time is not None or end_time is not None:
            return self._read_by_time(evaluator, start_time, end_time)
        if step_start is not None or step_end is not None:
            return self._read_by_step(evaluator, step_start, step_end)
        return self._read_by_eval(evaluator)
