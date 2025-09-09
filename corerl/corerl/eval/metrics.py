import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, NamedTuple, Protocol, SupportsFloat

import pandas as pd
from lib_config.config import config
from lib_sql.connection import TryConnectContextManager
from lib_sql.utils import SQLColumn, create_tsdb_table_query
from lib_utils.dict import flatten_tree, map_keys
from lib_utils.time import now_iso
from pydantic import Field
from sqlalchemy import text

from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig, WatermarkSyncConfig

log = logging.getLogger(__name__)


class _MetricPoint(NamedTuple):
    time: str
    agent_step : int
    metric: str
    value: float

class MetricsWriterProtocol(Protocol):
    def write(self, agent_step: int, metric: str, value: SupportsFloat, timestamp: str | None = None) -> None:
        ...

    def write_dict(
        self,
        values: Mapping[str, SupportsFloat | Mapping[str, Any]],
        agent_step: int,
        prefix: str = '',
    ) -> None:
        ...

    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        ...


    def close(self) -> None:
        ...


class DummyMetricsWriter:
    def write(self, agent_step: int | None, metric: str, value: SupportsFloat, timestamp: str | None = None) -> None:
        ...

    def write_dict(
        self,
        values: Mapping[str, SupportsFloat | Mapping[str, Any]],
        agent_step: int,
        prefix: str = '',
    ) -> None:
        ...

    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def close(self) -> None:
        ...


def create_metrics_writer(cfg: 'MetricsDBConfig') -> MetricsWriterProtocol:
    if not cfg.enabled:
        return DummyMetricsWriter()
    return MetricsTable(cfg)


@config()
class MetricsDBConfig(BufferedWriterConfig):
    table_name: str = 'metrics'
    enabled: bool = False
    narrow_format: bool = True
    watermark_cfg: WatermarkSyncConfig = Field(
        default_factory=lambda: WatermarkSyncConfig('watermark', 1, 256),
    )

class MetricsTable(BufferedWriter[_MetricPoint]):
    def __init__(
        self,
        cfg: MetricsDBConfig,
    ):
        super().__init__(cfg)
        self.cfg = cfg

        if not cfg.narrow_format:
            self.cfg.static_columns = False

    def _create_table_sql(self):
        if self.cfg.narrow_format:
            return create_tsdb_table_query(
                schema=self.cfg.table_schema,
                table=self.cfg.table_name,
                columns=[
                    SQLColumn(name='time', type='TIMESTAMP WITH TIME ZONE', nullable=False),
                    SQLColumn(name='agent_step', type='INTEGER', nullable=False),
                    SQLColumn(name='metric', type='TEXT', nullable=False),
                    SQLColumn(name='value', type='FLOAT', nullable=False),
                ],
                partition_column='metric',
                index_columns=['metric'],
            )
        return create_tsdb_table_query(
            schema=self.cfg.table_schema,
            table=self.cfg.table_name,
            columns=[
                SQLColumn(name='time', type='TIMESTAMP WITH TIME ZONE', nullable=False),
                SQLColumn(name='agent_step', type='INTEGER', nullable=False),
            ],
            partition_column=None,
            index_columns=[],
        )

    def write(self, agent_step: int | None, metric: str, value: SupportsFloat, timestamp: str | None = None):
        if agent_step is None:
            return

        if not self.cfg.enabled:
            return

        point = _MetricPoint(
            time=timestamp or now_iso(),
            agent_step=agent_step,
            metric=metric,
            value=float(value),
        )

        try:
            self._write(point)
        except Exception:
            log.exception(f'Failed to write metric: {metric} {value}')

    def _execute_read(self, stmt: str) -> pd.DataFrame:
        assert self.engine is not None
        with TryConnectContextManager(self.engine) as connection:
            return pd.read_sql(sql=text(stmt), con=connection)


    def _read_by_metric(self, metric: str) -> pd.DataFrame:
        if self.cfg.narrow_format:
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
            df.drop(columns=['value'], inplace=True)
            return df

        # wide format
        stmt = f"""
            SELECT
                time,
                agent_step,
                {metric}
            FROM {self.cfg.table_name}
        """
        df = self._execute_read(stmt)
        df["time"] = pd.to_datetime(df["time"])
        df["agent_step"] = df["agent_step"].astype(int)
        df[metric] = df[metric].astype(float)

        return df


    def _read_by_step(
        self,
        metric: str,
        step_start: int | None,
        step_end: int | None,
    ) -> pd.DataFrame:

        if self.cfg.narrow_format:
            stmt = f"""
            SELECT
                agent_step,
                value
            FROM {self.cfg.table_name}
            WHERE
                metric='{metric}'
            """

        else:
            stmt = f"""
                SELECT
                    agent_step,
                    {metric}
                FROM {self.cfg.table_name}
                WHERE 1=1
            """

        # step filtering logic the same whether narrow or wide
        if step_start is not None:
            stmt += f" AND agent_step>='{step_start}'"

        if step_end is not None:
            stmt += f" AND agent_step<='{step_end}'"

        stmt += ";"

        df = self._execute_read(stmt)
        df["agent_step"] = df["agent_step"].astype(int)

        if self.cfg.narrow_format:
            df[metric] = df["value"].astype(float)
            df.drop(columns=['value'], inplace=True)
            return df

        # wide format
        df[metric] = df[metric].astype(float)
        return df


    def _read_by_time(
        self,
        metric: str,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> pd.DataFrame:
        if self.cfg.narrow_format:
            stmt = f"""
                SELECT
                    time,
                    value
                FROM {self.cfg.table_name}
                WHERE
                    metric='{metric}'
            """
        else:
            stmt = f"""
                SELECT
                    time,
                    {metric}
                FROM {self.cfg.table_name}
                WHERE 1=1
            """

        # time filtering logic the same whether narrow or wide
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

        if self.cfg.narrow_format:
            df[metric] = df["value"].astype(float)
            df.drop(columns=['value'], inplace=True)
        else:
            df[metric] = df[metric].astype(float)

        return df


    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        # Make sure all MetricPoint objects in buffer have been written to DB
        self.blocking_sync()

        if start_time is not None or end_time is not None:
            return self._read_by_time(metric, start_time, end_time)
        if step_start is not None or step_end is not None:
            return self._read_by_step(metric, step_start, step_end)
        return self._read_by_metric(metric)

    def write_dict(
        self,
        values: Mapping[str, SupportsFloat | Mapping[str, Any]],
        prefix: str = '',
        agent_step: int | None = None,
    ) -> None:
        flattened = flatten_tree(values, prefix)
        for key, value in flattened.items():
            self.write(
                agent_step=agent_step,
                metric=key,
                value=value,
            )


    def _transform(self, points: list[_MetricPoint]):
        if self.cfg.narrow_format:
            return [point._asdict() for point in points]

        dict_points = [point._asdict() for point in points]

        time_as = itr_utils.keep_iterable(dict_points, keys=['time', 'agent_step'])
        time_as = itr_utils.group_by(time_as)
        metric_value =  itr_utils.keep_iterable(dict_points, keys=['metric', 'value'])
        metric_value = itr_utils.group_by_key(metric_value, 'metric', 'value')

        grouped = {**time_as, **metric_value}

        max_cols = { 'time', 'agent_step' }
        other_cols = set(grouped.keys()) - max_cols
        aggregated = (
            map_keys(grouped, max_cols, max)
            | map_keys(grouped, other_cols, lambda data: sum(data) / len(data))
        )
        return [aggregated]
