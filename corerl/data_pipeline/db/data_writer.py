import json
import logging
from datetime import UTC, datetime
from typing import NamedTuple

from sqlalchemy import text

from corerl.configs.config import config
from corerl.data_pipeline.tag_config import Agg
from corerl.sql_logging.sql_logging import get_sql_engine
from corerl.sql_logging.utils import SQLColumn, create_tsdb_table_query
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig

logger = logging.getLogger(__name__)

@config()
class TagDBConfig(BufferedWriterConfig):
    table_name: str = "sensors"
    data_agg: Agg = Agg.avg
    wide_format: bool = False


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
        self.engine = get_sql_engine(db_data=cfg, db_name=cfg.db_name)

    def write(
        self,
        timestamp: datetime,
        name: str,
        val: float | int | bool | str | None,
        host: str | None = None,
        id: str | None = None
    ) -> None:
        assert timestamp.tzinfo == UTC
        ts_iso = timestamp.isoformat()

        jsonb = json.dumps({"val": val})
        point = Point(
            ts=ts_iso,
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
        return create_tsdb_table_query(
            schema=self.cfg.table_schema,
            table=self.cfg.table_name,
            columns=[
                SQLColumn(name='time', type='TIMESTAMP WITH TIME ZONE', nullable=False),
                SQLColumn(name='host', type='TEXT', nullable=True),
                SQLColumn(name='id', type='TEXT', nullable=True),
                SQLColumn(name='name', type='TEXT', nullable=True),
                SQLColumn(name='fields', type='jsonb', nullable=True),
            ],
            partition_column='name',
            index_columns=['name'],
            chunk_time_interval='7d',
        )

    def flush(self) -> None:
        self.blocking_sync()
