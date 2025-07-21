from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

from lib_config.config import MISSING, computed, config
from lib_utils.sql_logging.connect_engine import TryConnectContextManager
from lib_utils.sql_logging.sql_logging import get_sql_engine, table_exists
from pydantic import Field
from sqlalchemy import Engine, TextClause

from corerl.sql_logging.sql_logging import SQLEngineConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


logger = logging.getLogger(__name__)


class SyncCond(Protocol):
    enabled: bool

    def is_soft_sync(self, writer: BufferedWriter)-> bool:
        ...
    def is_hard_sync(self, writer: BufferedWriter)-> bool:
        ...


@config()
class WatermarkSyncConfig:
    # name: Literal['watermark'] = 'watermark'
    enabled: bool = False
    lo_wm: int = 1024
    hi_wm: int = 2048


class WatermarkCond:
    def __init__(self, cfg: WatermarkSyncConfig):
        self.enabled = cfg.enabled
        self._lo_wm = cfg.lo_wm
        self._hi_wm = cfg.hi_wm

    def is_soft_sync(self, writer: 'BufferedWriter'):
        return len(writer) > self._lo_wm

    def is_hard_sync(self, writer: 'BufferedWriter'):
        return len(writer) > self._hi_wm


@config()
class BufferedWriterConfig(SQLEngineConfig):
    db_name: str = MISSING
    table_name: str = MISSING
    enabled: bool = True
    table_schema: str = MISSING

    # sync conditions
    sync_conds: list[str] = Field(default_factory=lambda:['watermark'])
    watermark_cfg: WatermarkSyncConfig = Field(default_factory=WatermarkSyncConfig)


    @computed('table_schema')
    @classmethod
    def _table_schema(cls, cfg: 'MainConfig'):
        return cfg.infra.db.schema

    @computed('db_name')
    @classmethod
    def _dbname(cls, cfg: 'MainConfig'):
        return cfg.infra.db.db_name


class BufferedWriter[T: NamedTuple](ABC):
    def __init__(
        self,
        cfg: BufferedWriterConfig,
        low_watermark: int = 1024,
        high_watermark: int = 2048,
    ) -> None:
        self.cfg = cfg

        self._low_wm = low_watermark
        self._hi_wm = high_watermark

        self._buffer: list[T] = []

        self._exec = ThreadPoolExecutor(max_workers=1)
        self._write_future: Future | None = None
        self.engine: Engine | None = None

        self._sync_conds: list[SyncCond] = []
        self._sync_conds.append(WatermarkCond(cfg.watermark_cfg))
        # prune out disabled conditions
        self._sync_conds = [cond for cond in self._sync_conds if cond.enabled]

        if self.cfg.enabled:
            self.engine = get_sql_engine(db_data=self.cfg, db_name=self.cfg.db_name)
            if not table_exists(self.engine, table_name=cfg.table_name, schema=cfg.table_schema):
                with TryConnectContextManager(self.engine) as connection:
                    connection.execute(self._create_table_sql())
                    connection.commit()


    def __len__(self):
        return len(self._buffer)


    def is_soft_sync(self):
        cond_results = [cond.is_soft_sync(self) for cond in self._sync_conds]
        return any(cond_results)


    def is_hard_sync(self):
        cond_results = [cond.is_hard_sync(self) for cond in self._sync_conds]
        return any(cond_results)


    @abstractmethod
    def _insert_sql(self) -> TextClause:
        ...


    @abstractmethod
    def _create_table_sql(self) -> TextClause:
        ...


    def _write(self, data: T) -> None:
        if not self.cfg.enabled:
            return

        self._buffer.append(data)

        if self.is_hard_sync():
            logger.warning('Buffer reached high watermark')
            # forcibly pause main thread until writer is finished
            assert self._write_future is not None
            self._write_future.result()

            # kick off a new background sync, since buffer is full
            self.background_sync()

        elif self.is_soft_sync():
            self.background_sync()


    def background_sync(self):
        if not self.cfg.enabled:
            return

        if self.is_writing():
            return

        data = self._buffer
        self._buffer = []
        self._write_future = self._exec.submit(self._deferred_write, data)


    def blocking_sync(self):
        if not self.cfg.enabled:
            return

        # wrap up in-progress sync
        if self._write_future is not None:
            self._write_future.result()

        self.background_sync()

        if self._write_future is not None:
            self._write_future.result()


    def is_writing(self):
        return self._write_future is not None and not self._write_future.done()


    def close(self) -> None:
        self.blocking_sync()
        self._exec.shutdown()


    def _deferred_write(self, points: list[T]):
        if len(points) == 0:
            return

        if not self.cfg.enabled:
            return

        assert self.engine is not None

        with TryConnectContextManager(self.engine) as connection:
            connection.execute(
                self._insert_sql(),
                [point._asdict() for point in points],
            )
            connection.commit()
