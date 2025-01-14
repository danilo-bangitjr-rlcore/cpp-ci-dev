import logging

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, UTC
from corerl.sql_logging.sql_logging import get_sql_engine, table_exists
from sqlalchemy import text, Engine
from corerl.data_pipeline.db.utils import try_connect
from corerl.data_pipeline.db.data_reader import TagDBConfig
from typing import NamedTuple


logger = logging.getLogger(__name__)


class Point(NamedTuple):
    ts: str
    name: str
    jsonb: str
    host: str
    id: str


class DataWriter:
    def __init__(
        self,
        db_cfg: TagDBConfig,
        low_watermark: int = 1024,
        high_watermark: int = 2048,
    ) -> None:
        self.engine: Engine = get_sql_engine(db_data=db_cfg, db_name=db_cfg.db_name)
        self.sensor_table_name = db_cfg.sensor_table_name
        self.host = "localhost"
        self.connection = try_connect(self.engine)

        self._low_wm = low_watermark
        self._hi_wm = high_watermark
        self._buffer: list[Point] = []

        self._exec = ThreadPoolExecutor(max_workers=1)
        self._write_future: Future | None = None

        self._has_built = False

    def _init(self):
        if self._has_built:
            return

        maybe_create_sensor_table(self.engine, self.sensor_table_name)
        self._has_built = True

    def write(
        self,
        timestamp: datetime,
        name: str,
        val: float,
        host: str | None = None,
        id: str | None = None
    ) -> None:
        self._init()
        assert timestamp.tzinfo == UTC

        # truncate microseconds
        timestamp = timestamp.replace(microsecond=0)

        self._buffer.append(Point(
            ts=timestamp.isoformat(),
            name=name,
            jsonb=f'{{"val": {val}}}',
            host=host or self.host,
            id=id or name,
        ))

        if len(self._buffer) > self._hi_wm:
            logger.warning('Buffer reached high watermark')
            # forcibly pause main thread until writer is finished
            assert self._write_future is not None
            self._write_future.result()

            # kick off a new background sync, since buffer is full
            self.background_sync()

        elif len(self._buffer) > self._low_wm:
            self.background_sync()


    def background_sync(self):
        if self.is_writing():
            return

        # swap out buffer pointer to start accumulating in new buffer
        data = self._buffer
        self._buffer = []
        self._write_future = self._exec.submit(self._write, data)


    def blocking_sync(self):
        # wrap up in-progress sync
        if self._write_future is not None:
            self._write_future.result()

        self.background_sync()
        assert self._write_future is not None
        self._write_future.result()


    def is_writing(self):
        return self._write_future is not None and not self._write_future.done()


    def _write(self, points: list[Point]):
        if len(points) == 0:
            return

        insert_stmt = f"""
            INSERT INTO {self.sensor_table_name}
            (time, host, id, name, fields)
            VALUES (TIMESTAMP :ts, :host, :id, :name, :jsonb);
        """

        self.connection.execute(
            text(insert_stmt),
            [point._asdict() for point in points]
        )
        self.connection.commit()

    def close(self) -> None:
        self.blocking_sync()
        self.connection.close()
        self._exec.shutdown()


def maybe_create_sensor_table(engine: Engine, sensor_table_name: str):
    if table_exists(engine, table_name=sensor_table_name):
        return

    create_table_stmt = f"""
        CREATE TABLE public.{sensor_table_name} (
            "time" timestamp with time zone NOT NULL,
            host text,
            id text,
            name text,
            fields jsonb
        );

        SELECT create_hypertable('{sensor_table_name}', 'time', chunk_time_interval => INTERVAL '7d');
        CREATE INDEX name_idx ON {sensor_table_name} (name);
        ALTER TABLE {sensor_table_name} SET (
            timescaledb.compress,
            timescaledb.compress_segmentby='name'
        );
    """

    with engine.connect() as con:
        con.execute(text(create_table_stmt))
        con.commit()
