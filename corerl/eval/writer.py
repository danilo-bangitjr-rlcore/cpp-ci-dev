import logging
import os
from collections import defaultdict
from typing import Literal, NamedTuple, Protocol, SupportsFloat

import pandas as pd
from pydantic import Field
from sqlalchemy import Connection, Engine, text
from typing_extensions import Annotated

from corerl.configs.config import config
from corerl.configs.group import Group
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig
from corerl.utils.time import now_iso

log = logging.getLogger(__name__)


class MetricsWriterProtocol(Protocol):
    def write(self, metric: str, value: SupportsFloat):
        ...
    def close(self)-> None:
        ...


class _MetricPoint(NamedTuple):
    timestamp: str
    metric: str
    value: float


@config()
class MetricsDBConfig(BufferedWriterConfig):
    name : Literal['db'] = 'db'
    db_name: str = 'postgres'
    table_name: str = 'metrics'
    table_schema: str = 'public'
    lo_wm: int = 128
    enabled: bool = False


class MetricsWriter(BufferedWriter[_MetricPoint]):
    def __init__(
        self,
        cfg: MetricsDBConfig,
        high_watermark: int = 256,
    ):
        super().__init__(cfg, cfg.lo_wm, high_watermark)
        self.cfg = cfg
        self._has_built = False

        self._eng: Engine | None = None
        self._conn: Connection | None = None


    def _create_table_sql(self):
        return text(f"""
            CREATE TABLE {self.cfg.table_schema}.{self.cfg.table_name} (
                time TIMESTAMP WITH time zone NOT NULL,
                metric text NOT NULL,
                value float NOT NULL
            );
            SELECT create_hypertable('{self.cfg.table_name}', 'time', chunk_time_interval => INTERVAL '1d');
            CREATE INDEX metric_idx ON {self.cfg.table_name} (metric);
            ALTER TABLE {self.cfg.table_name} SET (
                timescaledb.compress,
                timescaledb.compress_segmentby='metric'
            );
        """)


    def _insert_sql(self):
        return text(f"""
            INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
            (time, metric, value)
            VALUES (TIMESTAMP :timestamp, :metric, :value)
        """)


    def write(self, metric: str, value: SupportsFloat):
        if not self.cfg.enabled:
            return

        point = _MetricPoint(
            timestamp=now_iso(),
            metric=metric,
            value=float(value),
        )

        try:
            self._write(point)
        except Exception:
            log.exception(f'Failed to write metric: {metric} {value}')


@config()
class PandasMetricsConfig:
    name : Literal['pandas'] = 'pandas'
    output_path : str = 'metric_outputs'


class PandasMetricsWriter():
    def __init__(
            self,
            cfg : PandasMetricsConfig
        ):
        self.points = defaultdict(list)
        self.output_path = cfg.output_path
        os.makedirs(self.output_path, exist_ok=True)

    def write(self, metric: str, value: SupportsFloat):
        point = _MetricPoint(
            timestamp=now_iso(),
            metric=metric,
            value=float(value),
        )

        self.points[metric].append(point)

    def close(self):
        print("CLOSING")
        dfs = convert_to_dataframes(self.points)
        for metric, df in dfs.items():
            df.to_csv(f'{self.output_path}/{metric}.csv', index=False)


def convert_to_dataframes(metric_dict: dict[str, list[_MetricPoint]]) -> dict[str, pd.DataFrame]:
    dataframes = {}
    for metric, points in metric_dict.items():
        data = defaultdict(list)
        for point in points:
            data['timestamp'].append(point.timestamp)
            data['metric'].append(point.metric)
            data['value'].append(point.value)
        dataframes[metric] = pd.DataFrame(data)
    return dataframes


MetricsConfig = Annotated[
    MetricsDBConfig | PandasMetricsConfig,
    Field(discriminator='name')
    ]


metrics_group = Group[
    [], MetricsWriterProtocol,
]()

metrics_group.dispatcher(MetricsWriter)
metrics_group.dispatcher(PandasMetricsWriter)
