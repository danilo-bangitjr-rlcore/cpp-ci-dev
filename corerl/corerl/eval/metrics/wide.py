import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, SupportsFloat

import pandas as pd
from lib_sql.connection import TryConnectContextManager
from lib_sql.engine import get_sql_engine
from lib_sql.inspection import get_all_columns
from lib_sql.utils import SQLColumn, create_tsdb_table_query
from lib_sql.writers.buffered_sql_writer import BufferedSqlWriter
from lib_sql.writers.dynamic_schema_sql_writer import DynamicSchemaSqlWriter
from lib_sql.writers.point_collecting_sql_writer import PointCollectingSqlWriter
from lib_utils.dict import flatten_tree
from lib_utils.time import now_iso
from sqlalchemy import text

from corerl.eval.metrics.base import MetricsDBConfig

log = logging.getLogger(__name__)


class WideMetricsTable:
    """Metrics table implementation for wide format (time, agent_step, metric1, metric2, ...)."""

    # ============================================================================
    # Initialization & Configuration
    # ============================================================================

    def __init__(self, cfg: MetricsDBConfig):
        self.cfg = cfg
        self._current_agent_step: int | None = None

        self.engine = get_sql_engine(db_data=cfg, db_name=cfg.db_name)
        log.info(f"Created engine for database {cfg.db_name}")

        def table_factory(schema: str, table: str, columns: list[SQLColumn]):
            return create_tsdb_table_query(
                schema=schema,
                table=table,
                columns=columns,
                partition_column=None,
                index_columns=[],
            )

        initial_columns = [
            SQLColumn(name="time", type="TIMESTAMP WITH TIME ZONE", nullable=False),
            SQLColumn(name="agent_step", type="INTEGER", nullable=False),
        ]

        dynamic_writer = DynamicSchemaSqlWriter(
            engine=self.engine,
            table_name=cfg.table_name,
            table_creation_factory=table_factory,
            schema=cfg.table_schema,
            default_column_type="FLOAT",
            initial_columns=initial_columns,
        )

        buffered_writer = BufferedSqlWriter(
            inner=dynamic_writer,
            low_watermark=cfg.low_watermark,
            high_watermark=cfg.high_watermark,
            enabled=True,
        )

        def wide_row_factory(points: dict[str, Any]):
            timestamp = points.pop("__timestamp__", None)
            agent_step = points.pop("__agent_step__", None)

            if timestamp is None or agent_step is None:
                raise ValueError("Context not set for wide format write")

            return {
                "time": timestamp,
                "agent_step": agent_step,
                **{k: float(v) for k, v in points.items()},
            }

        self._writer = PointCollectingSqlWriter(
            inner=buffered_writer,
            row_factory=wide_row_factory,
            enabled=True,
        )

    def close(self):
        self._writer.close()

    # ============================================================================
    # Public Write API
    # ============================================================================

    def write(self, agent_step: int, metric: str, value: SupportsFloat, timestamp: str | None = None):
        if self._current_agent_step is not None and self._current_agent_step != agent_step:
            self._writer.collect_row()

        self._current_agent_step = agent_step
        current_time = datetime.fromisoformat(timestamp or now_iso())
        self._writer.write_point("__timestamp__", current_time)
        self._writer.write_point("__agent_step__", agent_step)

        return self._writer.write_point(metric, float(value))

    def write_dict(
        self,
        values: Mapping[str, SupportsFloat | Mapping[str, Any]],
        agent_step: int,
        prefix: str = "",
    ) -> None:
        flattened = flatten_tree(values, prefix)
        for key, value in flattened.items():
            self.write(
                agent_step=agent_step,
                metric=key,
                value=value,
            )

    def flush(self) -> None:
        self._writer.flush()

    # ============================================================================
    # Public Read API
    # ============================================================================

    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        # Make sure all data in buffer has been written to DB
        self._writer.flush()

        if start_time is not None or end_time is not None:
            return self._read_by_time(metric, start_time, end_time)
        if step_start is not None or step_end is not None:
            return self._read_by_step(metric, step_start, step_end)
        return self._read_by_metric(metric)

    # ============================================================================
    # Private Implementation
    # ============================================================================

    def _execute_read(self, stmt: str) -> pd.DataFrame:
        with TryConnectContextManager(self.engine) as connection:
            return pd.read_sql(sql=text(stmt), con=connection)

    def _get_matching_columns(self, metric: str, prefix_match: bool = False) -> list[str]:
        """Get column names that match the metric (exact or prefix)."""
        if prefix_match:
            columns = get_all_columns(self.engine, self.cfg.table_name)
            columns = [col['name'] for col in columns]

            return [
                col for col in columns
                if col.startswith(metric) and col not in {"time", "agent_step"}
            ]

        # Exact match - just return the metric if it exists
        return [metric]

    def _read_by_metric(self, metric: str, prefix_match: bool = False) -> pd.DataFrame:
        matching_columns = self._get_matching_columns(metric, prefix_match)

        if not matching_columns:
            # Return empty dataframe with expected structure
            return pd.DataFrame(columns=["time", "agent_step"])

        columns_str = ", ".join(matching_columns)
        stmt = f"""
            SELECT
                time,
                agent_step,
                {columns_str}
            FROM {self.cfg.table_name}
        """
        df = self._execute_read(stmt)
        df["time"] = pd.to_datetime(df["time"])
        df["agent_step"] = df["agent_step"].astype(int)

        for col in matching_columns:
            df[col] = df[col].astype(float)

        return df

    def _read_by_step(
        self,
        metric: str,
        step_start: int | None,
        step_end: int | None,
        prefix_match: bool = False,
    ) -> pd.DataFrame:
        matching_columns = self._get_matching_columns(metric, prefix_match)

        if not matching_columns:
            # Return empty dataframe with expected structure
            return pd.DataFrame(columns=["agent_step"])

        columns_str = ", ".join(matching_columns)
        stmt = f"""
            SELECT
                agent_step,
                {columns_str}
            FROM {self.cfg.table_name}
            WHERE 1=1
        """

        if step_start is not None:
            stmt += f" AND agent_step>='{step_start}'"

        if step_end is not None:
            stmt += f" AND agent_step<='{step_end}'"

        stmt += ";"

        df = self._execute_read(stmt)
        df["agent_step"] = df["agent_step"].astype(int)

        for col in matching_columns:
            df[col] = df[col].astype(float)

        return df

    def _read_by_time(
        self,
        metric: str,
        start_time: datetime | None,
        end_time: datetime | None,
        prefix_match: bool = False,
    ) -> pd.DataFrame:
        matching_columns = self._get_matching_columns(metric, prefix_match)

        if not matching_columns:
            # Return empty dataframe with expected structure
            return pd.DataFrame(columns=["time"])

        columns_str = ", ".join(matching_columns)
        stmt = f"""
            SELECT
                time,
                {columns_str}
            FROM {self.cfg.table_name}
            WHERE 1=1
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

        for col in matching_columns:
            df[col] = df[col].astype(float)

        return df
