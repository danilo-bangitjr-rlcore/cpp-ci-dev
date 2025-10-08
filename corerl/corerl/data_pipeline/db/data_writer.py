import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from lib_config.config import MISSING, computed, config
from lib_sql.engine import get_sql_engine
from lib_sql.utils import SQLColumn, create_tsdb_table_query
from lib_sql.writers.core.static_schema_sql_writer import StaticSchemaSqlWriter
from lib_sql.writers.transforms.buffered_sql_writer import BufferedSqlWriter

from corerl.sql_logging.sql_logging import SQLEngineConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)


@config()
class TagDBConfig(SQLEngineConfig):
    table_name: str = "sensors"
    wide_format: bool = False
    enabled: bool = True
    table_schema: str = MISSING
    db_name: str = MISSING

    low_watermark: int = 1024
    high_watermark: int = 2048

    @computed('table_schema')
    @classmethod
    def _table_schema(cls, cfg: 'MainConfig'):
        return cfg.infra.db.schema

    @computed('db_name')
    @classmethod
    def _dbname(cls, cfg: 'MainConfig'):
        return cfg.infra.db.db_name


class DataWriter:
    def __init__(
        self,
        cfg: TagDBConfig,
    ):
        self.cfg = cfg
        self.host = "localhost"

        self.engine = get_sql_engine(db_data=cfg, db_name=cfg.db_name)
        logger.info(f"Created engine for database {cfg.db_name}")

        def table_factory(schema: str, table: str, columns: list[SQLColumn]):
            return create_tsdb_table_query(
                schema=schema,
                table=table,
                columns=columns,
                partition_column='name',
                index_columns=['name'],
                chunk_time_interval='7d',
            )

        initial_columns = [
            SQLColumn(name='time', type='TIMESTAMP WITH TIME ZONE', nullable=False),
            SQLColumn(name='host', type='TEXT', nullable=True),
            SQLColumn(name='id', type='TEXT', nullable=True),
            SQLColumn(name='name', type='TEXT', nullable=True),
            SQLColumn(name='fields', type='jsonb', nullable=True),
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
            enabled=cfg.enabled,
        )

    def write(
        self,
        timestamp: datetime,
        name: str,
        val: float | bool | str | None,
        host: str | None = None,
        id: str | None = None,
    ) -> None:
        assert timestamp.tzinfo == UTC

        jsonb = json.dumps({"val": val})
        row = {
            'time': timestamp,
            'name': name,
            'fields': jsonb,
            'host': host or self.host,
            'id': id or name,
        }
        self._writer.write(row)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
