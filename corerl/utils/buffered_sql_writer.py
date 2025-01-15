import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generic, NamedTuple, TypeVar

from sqlalchemy import TextClause

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.db.utils import TryConnectContextManager
from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine, table_exists

logger = logging.getLogger(__name__)


@config()
class BufferedWriterConfig(SQLEngineConfig):
    db_name: str = 'postgres'
    table_name: str = MISSING


T = TypeVar('T', bound=NamedTuple)
class BufferedWriter(Generic[T], ABC):
    def __init__(
        self,
        cfg: BufferedWriterConfig,
        low_watermark: int = 1024,
        high_watermark: int = 2048,
    ) -> None:
        self.cfg = cfg
        self.table_name = cfg.table_name
        self.host = "localhost"

        self._low_wm = low_watermark
        self._hi_wm = high_watermark
        self._buffer: list[T] = []

        self._exec = ThreadPoolExecutor(max_workers=1)
        self._write_future: Future | None = None
        self.engine = get_sql_engine(db_data=self.cfg, db_name=self.cfg.db_name)

        if not table_exists(self.engine, table_name=self.table_name):
            with TryConnectContextManager(self.engine) as connection:
                connection.execute(self._create_table_sql())
                connection.commit()


    @abstractmethod
    def _insert_sql(self) -> TextClause:
        ...


    @abstractmethod
    def _create_table_sql(self) -> TextClause:
        ...


    def _write(self, data: T) -> None:
        self._buffer.append(data)

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
        self._write_future = self._exec.submit(self._deferred_write, data)


    def blocking_sync(self):
        # wrap up in-progress sync
        if self._write_future is not None:
            self._write_future.result()

        self.background_sync()
        assert self._write_future is not None
        self._write_future.result()


    def is_writing(self):
        return self._write_future is not None and not self._write_future.done()


    def close(self) -> None:
        self.blocking_sync()
        self._exec.shutdown()

    def _deferred_write(self, points: list[T]):
        if len(points) == 0:
            return

        with TryConnectContextManager(self.engine) as connection:
            connection.execute(
                self._insert_sql(),
                [point._asdict() for point in points]
            )
            connection.commit()
