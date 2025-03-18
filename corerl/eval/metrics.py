import logging
import os
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from typing import Literal, NamedTuple, Protocol, SupportsFloat

import pandas as pd
from pydantic import Field
from sqlalchemy import text
from typing_extensions import Annotated

from corerl.configs.config import config
from corerl.configs.group import Group
from corerl.data_pipeline.db.utils import TryConnectContextManager
from corerl.sql_logging.utils import SQLColumn, create_tsdb_table_query
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig
from corerl.utils.time import now_iso

log = logging.getLogger(__name__)


class MetricsTableProtocol(Protocol):
    def write(
        self,
        agent_step: int,
        metric: str,
        value: SupportsFloat,
        timestamp: str | None = None
    ):
        ...
    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> pd.DataFrame:
        ...
    def close(self)-> None:
        ...


class _MetricPoint(NamedTuple):
    timestamp: str
    agent_step : int
    metric: str
    value: float


@config()
class MetricsDBConfig(BufferedWriterConfig):
    name : Literal['db'] = 'db'
    table_name: str = 'metrics'
    lo_wm: int = 1
    enabled: bool = False

class MetricsTable(BufferedWriter[_MetricPoint]):
    def __init__(
        self,
        cfg: MetricsDBConfig,
        high_watermark: int = 256,
    ):
        super().__init__(cfg, cfg.lo_wm, high_watermark)
        self.cfg = cfg

    def _create_table_sql(self):
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


    def _insert_sql(self):
        return text(f"""
            INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
            (time, agent_step, metric, value)
            VALUES (TIMESTAMP WITH TIME ZONE :timestamp, :agent_step, :metric, :value)
        """)


    def write(self, agent_step: int, metric: str, value: SupportsFloat, timestamp: str | None = None):
        if not self.cfg.enabled:
            return

        point = _MetricPoint(
            timestamp=timestamp or now_iso(),
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
            metrics_table = pd.read_sql(sql=text(stmt), con=connection)

        return metrics_table

    def _read_by_metric(self, metric: str) -> pd.DataFrame:
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
        df["value"] = df["value"].astype(float)

        return df

    def _read_by_step(
        self,
        metric: str,
        step_start: int | None,
        step_end: int | None,
    ) -> pd.DataFrame:
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
        df["value"] = df["value"].astype(float)

        return df

    def _read_by_time(
        self,
        metric: str,
        start_time: datetime | None,
        end_time: datetime | None
    ) -> pd.DataFrame:
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
        df["value"] = df["value"].astype(float)

        return df

    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> pd.DataFrame:
        # Make sure all MetricPoint objects in buffer have been written to DB
        self.blocking_sync()

        if start_time is not None or end_time is not None:
            return self._read_by_time(metric, start_time, end_time)
        elif step_start is not None or step_end is not None:
            return self._read_by_step(metric, step_start, step_end)
        else:
            return self._read_by_metric(metric)


@config()
class PandasMetricsConfig:
    name : Literal['pandas'] = 'pandas'
    enabled: bool = False
    output_path : str = 'metric_outputs'
    buffer_size : int = 256  # Number of points to buffer before writing


class PandasMetricsTable:
    def __init__(
            self,
            cfg : PandasMetricsConfig,
        ):
        self.buffer = defaultdict(list)  # Temporary buffer for points
        self.output_path = cfg.output_path
        self.buffer_size = cfg.buffer_size

        if os.path.exists(self.output_path):
            logging.warning("Output path for metrics already exists. "
                            "Existing files will be overwritten.")
            shutil.rmtree(self.output_path)

        os.makedirs(self.output_path, exist_ok=True)

    def _read_by_metric(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        return df.loc[df['metric'] == metric].drop(columns=['metric'])

    def _read_by_step(
            self,
            df: pd.DataFrame,
            metric: str,
            step_start: int | None,
            step_end: int | None,
    ) -> pd.DataFrame:
        if step_start is not None:
            df = df.loc[df['agent_step'] >= step_start]

        if step_end is not None:
            df = df.loc[df['agent_step'] <= step_end]

        return df.loc[df['metric'] == metric].drop(columns=['metric', 'time'])

    def _read_by_time(
            self,
            df: pd.DataFrame,
            metric: str,
            start_time: datetime | None,
            end_time: datetime | None
    ) -> pd.DataFrame:
        if start_time is not None:
            df = df.loc[df['time'] >= start_time]

        if end_time is not None:
            df = df.loc[df['time'] <= end_time]

        return df.loc[df['metric'] == metric].drop(columns=['metric', 'agent_step'])

    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> pd.DataFrame:
        file_path = f'{self.output_path}/{metric}.csv'
        assert os.path.exists(file_path)
        df = pd.read_csv(file_path, header=0)
        df["time"] = pd.to_datetime(df["time"])
        df['agent_step'] = df['agent_step'].astype(int)
        df['metric'] = df['metric'].astype(str)
        df['value'] = df['value'].astype(float)

        if start_time is not None or end_time is not None:
            return self._read_by_time(df, metric, start_time, end_time)
        elif step_start is not None or step_end is not None:
            return self._read_by_step(df, metric, step_start, step_end)
        else:
            return self._read_by_metric(df, metric)

    def write(self, agent_step: int, metric: str, value: SupportsFloat, timestamp: str | None = None):
        point = _MetricPoint(
            timestamp=timestamp or now_iso(),
            agent_step=agent_step,
            metric=metric,
            value=float(value),
        )

        self.buffer[metric].append(point)

        if len(self.buffer[metric]) >= self.buffer_size:
            self._flush_metric(metric)

    def _flush_metric(self, metric: str):
        """Write buffered points for a specific metric to CSV"""
        if not self.buffer[metric]:
            return

        data = defaultdict(list)
        for point in self.buffer[metric]:
            data['time'].append(point.timestamp)
            data['agent_step'].append(point.agent_step)
            data['metric'].append(point.metric)
            data['value'].append(point.value)

        df = pd.DataFrame(data)
        file_path = f'{self.output_path}/{metric}.csv'

        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        self.buffer[metric].clear()

    def close(self):
        # Flush any remaining points
        for metric in list(self.buffer.keys()):
            self._flush_metric(metric)


MetricsConfig = Annotated[
    MetricsDBConfig | PandasMetricsConfig,
    Field(discriminator='name')
]


metrics_group = Group[
    [], MetricsTableProtocol,
]()

metrics_group.dispatcher(MetricsTable)
metrics_group.dispatcher(PandasMetricsTable)
