import logging
import warnings
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, SupportsFloat

import pandas as pd
from lib_sql.connection import TryConnectContextManager
from lib_sql.engine import get_sql_engine
from lib_sql.utils import SQLColumn, create_tsdb_table_query
from lib_sql.writers.core.static_schema_sql_writer import StaticSchemaSqlWriter
from lib_sql.writers.transforms.buffered_sql_writer import BufferedSqlWriter
from lib_utils.dict import flatten_tree
from sqlalchemy import text

from corerl.configs.eval.metrics import MetricsDBConfig

log = logging.getLogger(__name__)


class NarrowMetricsTable:
    """DEPRECATED: Metrics table implementation for narrow format (time, agent_step, metric, value).

    This implementation is deprecated and will be removed in a future version.
    Please use WideMetricsTable (narrow_format=False) instead.
    """

    # ============================================================================
    # Initialization & Configuration
    # ============================================================================

    def __init__(self, cfg: MetricsDBConfig, time_provider: Callable[[], datetime] | None = None):
        warnings.warn(
            "NarrowMetricsTable is deprecated and will be removed in a future version. "
            "Please use WideMetricsTable (narrow_format=False) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.cfg = cfg
        self._time_provider = time_provider or (lambda: datetime.now(UTC))

        self.engine = get_sql_engine(db_data=cfg, db_name=cfg.db_name)
        log.info(f"Created engine for database {cfg.db_name}")

        def table_factory(schema: str, table: str, columns: list[SQLColumn]):
            return create_tsdb_table_query(
                schema=schema,
                table=table,
                columns=columns,
                partition_column="metric",
                index_columns=["metric"],
            )

        initial_columns = [
            SQLColumn(name="time", type="TIMESTAMP WITH TIME ZONE", nullable=False),
            SQLColumn(name="agent_step", type="INTEGER", nullable=False),
            SQLColumn(name="metric", type="TEXT", nullable=False),
            SQLColumn(name="value", type="DOUBLE PRECISION", nullable=False),
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

    def close(self):
        self.flush()
        self._writer.close()

    @property
    def table_name(self) -> str:
        return self.cfg.table_name

    # ============================================================================
    # Public Write API
    # ============================================================================

    def write(self, agent_step: int, metric: str, value: SupportsFloat, timestamp: str | None = None):
        current_time = datetime.fromisoformat(timestamp) if timestamp else self._time_provider()
        return self._writer.write(
            {
                "time": current_time,
                "agent_step": agent_step,
                "metric": metric,
                "value": float(value),
            },
        )

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
        prefix_match: bool = False,
    ) -> pd.DataFrame:
        # Make sure all data in buffer has been written to DB
        self._writer.flush()

        if start_time is not None or end_time is not None:
            return self._read_by_time(metric, start_time, end_time, prefix_match)
        if step_start is not None or step_end is not None:
            return self._read_by_step(metric, step_start, step_end, prefix_match)
        return self._read_by_metric(metric, prefix_match)

    # ============================================================================
    # Private Implementation
    # ============================================================================

    def _execute_read(self, stmt: str) -> pd.DataFrame:
        with TryConnectContextManager(self.engine) as connection:
            return pd.read_sql(sql=text(stmt), con=connection)

    def _read_by_metric(self, metric: str, prefix_match: bool = False) -> pd.DataFrame:
        if prefix_match:
            stmt = f"""
                SELECT
                    time,
                    agent_step,
                    metric,
                    value
                FROM {self.cfg.table_name}
                WHERE
                    metric LIKE '{metric}%';
            """
            df = self._execute_read(stmt)
            df["time"] = pd.to_datetime(df["time"])
            df["agent_step"] = df["agent_step"].astype(int)

            # Pivot the data to create columns for each metric
            pivot_df = df.pivot_table(
                index=["time", "agent_step"],
                columns="metric",
                values="value",
                aggfunc="first",
            ).reset_index()

            # Flatten column names
            pivot_df.columns.name = None
            return pivot_df

        stmt = f"""
            SELECT
                time,
                agent_step,
                value
            FROM {self.cfg.table_name}
            WHERE
                metric='{metric}';
        """
        df = self._execute_read(stmt)
        df["time"] = pd.to_datetime(df["time"])
        df["agent_step"] = df["agent_step"].astype(int)
        df[metric] = df["value"].astype(float)
        df.drop(columns=["value"], inplace=True)
        return df

    def _read_by_step(
        self,
        metric: str,
        step_start: int | None,
        step_end: int | None,
        prefix_match: bool = False,
    ) -> pd.DataFrame:
        if prefix_match:
            stmt = f"""
            SELECT
                agent_step,
                metric,
                value
            FROM {self.cfg.table_name}
            WHERE
                metric LIKE '{metric}%'
            """
        else:
            stmt = f"""
            SELECT
                agent_step,
                value
            FROM {self.cfg.table_name}
            WHERE
                metric='{metric}'
            """

        if step_start is not None:
            stmt += f" AND agent_step>='{step_start}'"

        if step_end is not None:
            stmt += f" AND agent_step<='{step_end}'"

        stmt += ";"

        df = self._execute_read(stmt)
        df["agent_step"] = df["agent_step"].astype(int)

        if prefix_match:
            # Pivot the data to create columns for each metric
            pivot_df = df.pivot_table(
                index="agent_step",
                columns="metric",
                values="value",
                aggfunc="first",
            ).reset_index()
            # Flatten column names
            pivot_df.columns.name = None
            return pivot_df

        df[metric] = df["value"].astype(float)
        df.drop(columns=["value"], inplace=True)
        return df

    def _read_by_time(
        self,
        metric: str,
        start_time: datetime | None,
        end_time: datetime | None,
        prefix_match: bool = False,
    ) -> pd.DataFrame:
        if prefix_match:
            stmt = f"""
                SELECT
                    time,
                    metric,
                    value
                FROM {self.cfg.table_name}
                WHERE
                    metric LIKE '{metric}%'
            """
        else:
            stmt = f"""
                SELECT
                    time,
                    value
                FROM {self.cfg.table_name}
                WHERE
                    metric='{metric}'
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

        if prefix_match:
            # Pivot the data to create columns for each metric
            pivot_df = df.pivot_table(
                index="time",
                columns="metric",
                values="value",
                aggfunc="first",
            ).reset_index()
            # Flatten column names
            pivot_df.columns.name = None
            return pivot_df

        df[metric] = df["value"].astype(float)
        df.drop(columns=["value"], inplace=True)

        return df
