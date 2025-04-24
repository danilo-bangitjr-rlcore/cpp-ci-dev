import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NamedTuple

from sqlalchemy import text

from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.tag_config import Agg
from corerl.sql_logging.sql_logging import get_sql_engine
from corerl.sql_logging.utils import SQLColumn, create_tsdb_table_query
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)

@config()
class TagDBConfig(BufferedWriterConfig):
    table_name: str = "sensors"
    data_agg: Agg = Agg.avg
    table_schema: str = MISSING
    wide_format: bool = False

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
        self.engine = get_sql_engine(db_data=cfg, db_name=cfg.db_name)
        self.point = None
        self.val = None

    def write(
        self,
        timestamp: datetime,
        name: str,
        val: float,
        host: str | None = None,
        id: str | None = None
    ) -> None:
        assert timestamp.tzinfo == UTC

        if not self.cfg.wide_format:
            jsonb = json.dumps({"val": val})
            point = Point(
                ts=timestamp.isoformat(),
                name=name,
                jsonb=jsonb,
                host=host or self.host,
                id=id or name,
            )
        else:
            self.point = Point(
                ts=timestamp.isoformat(),
                name=name,
                jsonb="",
                host="",
                id="",
            )
            self.val = val
            return self._write_wide(timestamp, name, val)

        self._write(point)

    def _insert_sql(self):
        if not self.cfg.wide_format:
            return text(f"""
                INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
                (time, host, id, name, fields)
                VALUES (TIMESTAMP :ts, :host, :id, :name, :jsonb);
            """)
        else:
            return text(f"""
                INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
                (time, {self.point.name})
                VALUES (TIMESTAMP :ts, :val);
            """)

    def _create_table_sql(self):
        if not self.cfg.wide_format:
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
        else:
            # for wide format, we need a primary key on time for ON CONFLICT to work
            return text(f"""
                CREATE SCHEMA IF NOT EXISTS {self.cfg.table_schema};
                CREATE TABLE IF NOT EXISTS {self.cfg.table_schema}.{self.cfg.table_name} (
                    time TIMESTAMP WITH TIME ZONE NOT NULL PRIMARY KEY
                );
                SELECT create_hypertable('{self.cfg.table_schema}.{self.cfg.table_name}', 'time',
                                         if_not_exists => TRUE,
                                         chunk_time_interval => INTERVAL '7d');
                ALTER TABLE {self.cfg.table_schema}.{self.cfg.table_name} SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby='time'
                );
            """)

    def _write_wide(self, timestamp: datetime, name: str, val: float) -> None:
        if self.engine is None:
            self.engine = get_sql_engine(db_data=self.cfg, db_name=self.cfg.db_name)

        column_type = "FLOAT"
        if isinstance(val, bool):
            column_type = "BOOLEAN"

        with self.engine.connect() as conn:
            conn.execute(
                text(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT FROM information_schema.columns
                            WHERE table_schema = '{self.cfg.table_schema}'
                            AND table_name = '{self.cfg.table_name}'
                            AND column_name = '{name}'
                        ) THEN
                            ALTER TABLE {self.cfg.table_schema}.{self.cfg.table_name}
                            ADD COLUMN "{name}" {column_type};
                        END IF;
                    END $$;
                """)
            )

            conn.execute(
                text(f"""
                    INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name} (time, "{name}")
                    VALUES (TIMESTAMP :ts, :val)
                    ON CONFLICT (time) DO UPDATE SET "{name}" = :val;
                """),
                {"ts": timestamp.isoformat(), "val": val}
            )
            conn.commit()

    def flush(self) -> None:
        if not self.cfg.wide_format:
            self.blocking_sync()
        # for wide format, no buffering is used, so flush is a no-op
