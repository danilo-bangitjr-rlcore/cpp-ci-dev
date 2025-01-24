import json
import logging
from datetime import UTC, datetime
from typing import Literal, NamedTuple

from sqlalchemy import text

from corerl.configs.config import config
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig

logger = logging.getLogger(__name__)

@config()
class TagDBConfig(BufferedWriterConfig):
    db_name: str = "postgres"
    table_name: str = "sensors"
    table_schema: str = "public"
    data_agg: Literal["avg", "last", "bool_or"] = "avg"


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

        # truncate microseconds
        timestamp = timestamp.replace(microsecond=0)

        jsonb = json.dumps({"val": val}, allow_nan=False).lower()
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
            INSERT INTO {self.cfg.table_schema}.{self.table_name}
            (time, host, id, name, fields)
            VALUES (TIMESTAMP :ts, :host, :id, :name, :jsonb);
        """)


    def _create_table_sql(self):
        return text(f"""
            CREATE TABLE {self.cfg.table_schema}.{self.table_name} (
                "time" TIMESTAMP WITH time zone NOT NULL,
                host text,
                id text,
                name text,
                fields jsonb
            );

            SELECT create_hypertable('{self.table_name}', 'time', chunk_time_interval => INTERVAL '7d');
            CREATE INDEX name_idx ON {self.table_name} (name);
            ALTER TABLE {self.table_name} SET (
                timescaledb.compress,
                timescaledb.compress_segmentby='name'
            );
        """)
