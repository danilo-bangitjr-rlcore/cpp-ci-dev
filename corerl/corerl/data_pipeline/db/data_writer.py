import json
import logging
from datetime import UTC, datetime
from typing import NamedTuple

from lib_config.config import config
from lib_sql.engine import get_sql_engine
from lib_sql.utils import SQLColumn, create_tsdb_table_query

from corerl.tags.components.opc import Agg
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig

logger = logging.getLogger(__name__)

@config()
class TagDBConfig(BufferedWriterConfig):
    table_name: str = "sensors"
    data_agg: Agg = Agg.avg
    wide_format: bool = False


class Point(NamedTuple):
    time: str
    name: str
    fields: str
    host: str
    id: str


class DataWriter(BufferedWriter[Point]):
    def __init__(
        self,
        cfg: TagDBConfig,
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.host = "localhost"
        self.engine = get_sql_engine(db_data=cfg, db_name=cfg.db_name)

    def write(
        self,
        timestamp: datetime,
        name: str,
        val: float | bool | str | None,
        host: str | None = None,
        id: str | None = None,
    ) -> None:
        assert timestamp.tzinfo == UTC
        ts_iso = timestamp.isoformat()

        jsonb = json.dumps({"val": val})
        point = Point(
            time=ts_iso,
            name=name,
            fields=jsonb,
            host=host or self.host,
            id=id or name,
        )
        self._write(point)

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
