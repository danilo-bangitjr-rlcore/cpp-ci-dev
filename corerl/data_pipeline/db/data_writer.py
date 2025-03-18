import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NamedTuple

from sqlalchemy import text

from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.tag_config import Agg
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)

@config()
class TagDBConfig(BufferedWriterConfig):
    table_name: str = "sensors"
    data_agg: Agg = Agg.avg
    table_schema: str = MISSING

    @computed('table_schema')
    @classmethod
    def _table_schema(cls, cfg: 'MainConfig'):
        return cfg.infra.db.schema


class Point(NamedTuple):
    ts: str
    name: str
    jsonb: str
    host: str
    id: str


class DataWriter(BufferedWriter[Point]):
    def __init__(
        self,
        cfg: TagDBConfig,
        low_watermark: int = 1024,
        high_watermark: int = 2048,
    ):
        super().__init__(cfg, low_watermark, high_watermark)
        self.cfg = cfg
        self.host = "localhost"

    def write(
        self,
        timestamp: datetime,
        name: str,
        val: float,
        host: str | None = None,
        id: str | None = None
    ) -> None:
        assert timestamp.tzinfo == UTC

        jsonb = json.dumps({"val": val})
        point = Point(
            ts=timestamp.isoformat(),
            name=name,
            jsonb=jsonb,
            host=host or self.host,
            id=id or name,
        )

        self._write(point)


    def _insert_sql(self):
        return text(f"""
            INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
            (time, host, id, name, fields)
            VALUES (TIMESTAMP :ts, :host, :id, :name, :jsonb);
        """)


    def _create_table_sql(self):
        schema_builder = ''
        if self.cfg.table_schema != 'public':
            schema_builder = f'CREATE SCHEMA IF NOT EXISTS {self.cfg.table_schema};'

        table = self.cfg.table_schema + '.' + self.cfg.table_name
        return text(f"""
            {schema_builder}
            CREATE TABLE {table} (
                "time" TIMESTAMP WITH time zone NOT NULL,
                host text,
                id text,
                name text,
                fields jsonb
            );

            SELECT create_hypertable('{table}', 'time', chunk_time_interval => INTERVAL '7d');
            CREATE INDEX {self.cfg.table_name}_idx ON {table} (name);
            ALTER TABLE {table} SET (
                timescaledb.compress,
                timescaledb.compress_segmentby='name'
            );
        """)
