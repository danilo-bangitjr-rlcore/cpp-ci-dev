import logging
from typing import NamedTuple, SupportsFloat

from sqlalchemy import Connection, Engine, text

from corerl.configs.config import config
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig
from corerl.utils.time import now_iso

log = logging.getLogger(__name__)


class _MetricPoint(NamedTuple):
    timestamp: str
    metric: str
    value: float


@config()
class MetricsDBConfig(BufferedWriterConfig):
    db_name: str = 'postgres'
    table_name: str = 'metrics'
    table_schema: str = 'public'
    enabled: bool = False


class MetricsWriter(BufferedWriter[_MetricPoint]):
    def __init__(
        self,
        cfg: MetricsDBConfig,
        low_watermark: int = 128,
        high_watermark: int = 256,
    ):
        super().__init__(cfg, low_watermark, high_watermark)
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
