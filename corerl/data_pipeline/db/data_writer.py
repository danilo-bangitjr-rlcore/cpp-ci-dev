import json
import logging
from collections import namedtuple
from datetime import UTC, datetime
from typing import Any, NamedTuple, Protocol

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




class WideTagConfig(Protocol):
    name: str
    dtype: str


class WideDataWriter(BufferedWriter[NamedTuple]):
    def __init__(
        self,
        cfg: TagDBConfig,
        tag_cfgs: list[WideTagConfig],
        low_watermark: int = 1024,
        high_watermark: int = 2048,
    ):
        self._tag_cfgs = tag_cfgs
        super().__init__(cfg, low_watermark, high_watermark)
        self.cfg = cfg

        self._tag_names = [tag.name for tag in tag_cfgs]
        self._point_builder = namedtuple('Point', ['time'] + self._tag_names)

    def write(
        self,
        timestamp: datetime,
        tag_values: dict[str, Any],
    ) -> None:
        assert timestamp.tzinfo == UTC
        ts_iso = timestamp.isoformat()

        data = { 'time': ts_iso } | {
            name: tag_values.get(name)
            for name in self._tag_names
        }
        point = self._point_builder(**data)
        self._write(point)

    def _insert_sql(self):
        tag_name_strs = ','.join(self._tag_names)
        variable_bindings = ','.join(map(
            lambda n: f':{n}',
            self._tag_names,
        ))
        return text(f"""
            INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
            (time, {tag_name_strs})
            VALUES (TIMESTAMP :time, {variable_bindings});
        """)

    def _create_table_sql(self):
        return create_tsdb_table_query(
            schema=self.cfg.table_schema,
            table=self.cfg.table_name,
            columns=[
                SQLColumn(name='time', type='TIMESTAMP WITH TIME ZONE', nullable=False),
            ] + [
                SQLColumn(name=tag.name, type=tag.dtype, nullable=True)
                for tag in self._tag_cfgs
            ],
            partition_column='time',
            index_columns=[],
            chunk_time_interval='7d',
        )

    def flush(self) -> None:
        self.blocking_sync()
