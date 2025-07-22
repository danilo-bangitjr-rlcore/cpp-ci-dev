import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

from lib_config.config import MISSING, computed, config
from lib_config.group import Group
from lib_utils.sql_logging.connect_engine import TryConnectContextManager
from lib_utils.sql_logging.sql_logging import get_sql_engine, table_exists
from pydantic import Field
from sqlalchemy import Engine, TextClause, text

from corerl.sql_logging.sql_logging import SQLEngineConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


logger = logging.getLogger(__name__)

class SyncCond(Protocol):
    def is_soft_sync(self, writer: 'BufferedWriter') -> bool:
        ...
    def is_hard_sync(self, writer: 'BufferedWriter') -> bool:
        ...


@config()
class WatermarkSyncConfig:
    name: Literal['watermark'] = 'watermark'
    lo_wm: int = 1024
    hi_wm: int = 2048

class WatermarkCond:
    def __init__(self, cfg: WatermarkSyncConfig):
        self._lo_wm = cfg.lo_wm
        self._hi_wm = cfg.hi_wm

    def is_soft_sync(self, writer: 'BufferedWriter'):
        return len(writer) > self._lo_wm

    def is_hard_sync(self, writer: 'BufferedWriter'):
        return len(writer) > self._hi_wm


@config()
class TimeSyncConfig:
    name: Literal['time'] = 'time'
    soft_sync_seconds: int = 5
    hard_sync_seconds: int = 10

class TimeSyncCond:
    def __init__(self, cfg: TimeSyncConfig):
        self._soft_sync_seconds = cfg.soft_sync_seconds
        self._hard_sync_seconds = cfg.hard_sync_seconds

    def is_soft_sync(self, writer: 'BufferedWriter'):
        current_time = time.time()
        time_elapsed = current_time - writer.last_sync_time
        return time_elapsed >= self._soft_sync_seconds

    def is_hard_sync(self, writer: 'BufferedWriter'):
        current_time = time.time()
        time_elapsed = current_time - writer.last_sync_time
        return time_elapsed >= self._hard_sync_seconds


sync_group = Group[
    [], SyncCond,
]()
sync_group.dispatcher(WatermarkCond)
sync_group.dispatcher(TimeSyncCond)


@config()
class BufferedWriterConfig(SQLEngineConfig):
    db_name: str = MISSING
    table_name: str = MISSING
    enabled: bool = True
    table_schema: str = MISSING

    # sync conditions
    sync_conds: list[str] = Field(default_factory=lambda: ['watermark'])
    watermark_cfg: WatermarkSyncConfig = Field(default_factory=WatermarkSyncConfig)
    timesync_cfg: TimeSyncConfig = Field(default_factory=TimeSyncConfig)

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
    ) -> None:
        self.cfg = cfg

        self._buffer: list[T] = []

        self._exec = ThreadPoolExecutor(max_workers=1)
        self._write_future: Future | None = None
        self.engine: Engine | None = None
        self._columns_initialized = False
        self._known_columns: set[str] = set()

        self._sync_conds: list[SyncCond] = []
        self.last_sync_time = time.time()

        # Mapping of condition names to their condfigs
        cond_cfg_map = {
            'watermark': cfg.watermark_cfg,
            'time': cfg.timesync_cfg,
        }

        for cond_name in cfg.sync_conds:
            cond_cfg = cond_cfg_map.get(cond_name)
            assert cond_cfg is not None, "Invalid sync condition specified."
            sync_cond = sync_group.dispatch(cond_cfg)
            self._sync_conds.append(sync_cond)

        if self.cfg.enabled:
            self.engine = get_sql_engine(db_data=self.cfg, db_name=self.cfg.db_name)

    def _ensure_table_exists(self):
        assert self.engine is not None
        if not table_exists(self.engine, table_name=self.cfg.table_name, schema=self.cfg.table_schema):
            with TryConnectContextManager(self.engine) as connection:
                # Create new table
                connection.execute(self._create_table_sql())
                connection.commit()
            self._table_created = True


    def _ensure_known_columns_initialized(self):
        if not self._columns_initialized and self.engine is not None:
            with TryConnectContextManager(self.engine) as connection:
                result = connection.execute(self._get_columns_sql())
                self._known_columns = {row[0].strip('"') for row in result}
            self._columns_initialized = True


    def __len__(self):
        return len(self._buffer)


    def is_soft_sync(self):
        return any(cond.is_soft_sync(self) for cond in self._sync_conds)

    def is_hard_sync(self):
        return any(cond.is_hard_sync(self) for cond in self._sync_conds)

    @abstractmethod
    def _create_table_sql(self) -> TextClause:
        ...

    def _get_columns_sql(self) -> TextClause:
        return text(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = '{self.cfg.table_schema}'
                AND table_name = '{self.cfg.table_name}'
            """)

    def _add_column_sql(self, col_name: str):
        return text(f"""
            ALTER TABLE {self.cfg.table_schema}.{self.cfg.table_name}
            ADD COLUMN IF NOT EXISTS "{col_name}" FLOAT
        """)

    def _write(self, data: T) -> None:
        if not self.cfg.enabled:
            return

        self._buffer.append(data)

        if self.is_hard_sync():
            logger.warning('Hard sync condition reached')
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
        self.last_sync_time = time.time()
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


    def _get_columns(self, dict_points: list[dict]):
        points_columns = set()
        for point in dict_points:
            points_columns.update(k for k, _ in point.items())
        return sorted(points_columns)


    def _add_columns(self, points_columns: list):
        if not points_columns:
            return

        new_columns = [col for col in points_columns
                       if col not in self._known_columns]

        assert self.engine is not None

        for column in new_columns:
            with TryConnectContextManager(self.engine) as connection:
                connection.execute(self._add_column_sql(column))
                connection.commit()
            self._known_columns.add(column)


    def _sanitize_keys(self, dict_points: list[dict]):
        def _sanitize_key(name: str):
            # remove non alpha-numeric characters and spaces
            sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
            # Replace multiple consecutive underscores with single underscore
            return re.sub(r'_+', '_', sanitized)

        def _sanitize_dict_keys(d: dict[str, Any]):
            keys = list(d.keys())
            for key in keys:
                sanitized_key = _sanitize_key(key)
                if sanitized_key != key:
                    d[sanitized_key] = d.pop(key)

        # Sanitize the dictionary keys
        for point in dict_points:
            _sanitize_dict_keys(point)

        return dict_points

    def _insert_sql(self, columns: list):
        # Create dynamic INSERT statement with only columns that have data, will default to null
        columns_list = ", ".join(f'"{col}"' for col in sorted(columns))
        placeholders = ", ".join(f":{col}" for col in sorted(columns))

        sql = f"""
            INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
            ({columns_list})
            VALUES ({placeholders})
        """
        return text(sql)

    def _deferred_write(self, points: list[T]):
        if len(points) == 0:
            return

        if not self.cfg.enabled:
            return

        assert self.engine is not None

        self._ensure_table_exists()
        self._ensure_known_columns_initialized()

        dict_points = [point._asdict() for point in points]
        dict_points = self._sanitize_keys(dict_points)
        points_columns = self._get_columns(dict_points)
        self._add_columns(points_columns)

        with TryConnectContextManager(self.engine) as connection:
            connection.execute(
                self._insert_sql(points_columns),
                dict_points,
            )
            connection.commit()
