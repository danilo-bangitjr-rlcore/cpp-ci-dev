import logging

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, UTC
from corerl.sql_logging.sql_logging import get_sql_engine, table_exists, SQLEngineConfig
from sqlalchemy import text, Engine
from corerl.data_loaders.utils import try_connect
from typing import NamedTuple


logger = logging.getLogger(__name__)


class Point(NamedTuple):
    ts: str
    name: str
    jsonb: str
    host: str
    id: str
    quality: str


class DataWriter:
    def __init__(
        self,
        db_cfg: SQLEngineConfig,
        db_name: str,
        sensor_table_name: str,
        low_watermark: int = 1024,
        high_watermark: int = 2048,
    ) -> None:
        self.engine: Engine = get_sql_engine(db_data=db_cfg, db_name=db_name)
        self.sensor_table_name = sensor_table_name
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
        id: str | None = None,
        quality: str | None = None,
    ) -> None:
        self._init()
        assert timestamp.tzinfo == UTC

        self._buffer.append(Point(
            ts=timestamp.isoformat(),
            name=name,
            jsonb=f'{{"val": {val}}}',
            host=host or self.host,
            id=id or name,
            quality=quality or "The operation succeeded. StatusGood (0x0)",
        ))

        if len(self._buffer) > self._hi_wm:
            # forcibly pause main thread until writer is finished
            assert self._write_future is not None
            self._write_future.result()
            logger.warning('Buffer reached high watermark')

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
        insert_stmt = f"""
            INSERT INTO {self.sensor_table_name}
            (time, host, id, name, \"Quality\", fields)
            VALUES (TIMESTAMP :ts, :host, :id, :name, :quality, :jsonb);
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
            "Quality" text,
            fields jsonb
        );
    """
    create_hypertable_stmt = f"""
        SELECT create_hypertable('{sensor_table_name}', 'time', chunk_time_interval => INTERVAL '1h');
    """
    with engine.connect() as con:
        con.execute(text(create_table_stmt))
        con.execute(text(create_hypertable_stmt))
        con.commit()
