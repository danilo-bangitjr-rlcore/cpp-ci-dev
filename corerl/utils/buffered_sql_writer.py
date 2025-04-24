from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, Generic, NamedTuple, TypeVar

from sqlalchemy import Engine, TextClause, text
from sqlalchemy.engine import Connection

from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.db.utils import TryConnectContextManager
from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine, table_exists

if TYPE_CHECKING:
    from corerl.config import MainConfig


logger = logging.getLogger(__name__)


@config()
class BufferedWriterConfig(SQLEngineConfig):
    db_name: str = MISSING
    table_name: str = MISSING
    enabled: bool = True
    table_schema: str = MISSING
    wide_format: bool = False

    @computed('table_schema')
    @classmethod
    def _table_schema(cls, cfg: MainConfig):
        return cfg.infra.db.schema

    @computed('db_name')
    @classmethod
    def _dbname(cls, cfg: MainConfig):
        return cfg.infra.db.db_name


T = TypeVar('T', bound=NamedTuple)
class BufferedWriter(Generic[T], ABC):
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
        self._wide_buffer: Dict[str, Dict[str, Any]] = {}

        self._exec = ThreadPoolExecutor(max_workers=1)
        self._write_future: Future | None = None
        self.engine: Engine | None = None

        if self.cfg.enabled:
            self.engine = get_sql_engine(db_data=self.cfg, db_name=self.cfg.db_name)
            if not table_exists(self.engine, table_name=cfg.table_name, schema=cfg.table_schema):
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
        if not self.cfg.enabled:
            return

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


    def _write_wide(self, timestamp: str, name: str, value: Any) -> None:
        if not self.cfg.enabled:
            return

        if timestamp not in self._wide_buffer:
            self._wide_buffer[timestamp] = {}

        self._wide_buffer[timestamp][name] = value
        if len(self._wide_buffer) > self._hi_wm:
            logger.warning('Wide buffer reached high watermark')
            if self._write_future is not None:
                self._write_future.result()
            self.background_sync_wide()
        elif len(self._wide_buffer) > self._low_wm:
            self.background_sync_wide()


    def background_sync(self):
        if not self.cfg.enabled:
            return

        if self.is_writing():
            return

        # swap out buffer pointer to start accumulating in new buffer
        data = self._buffer
        self._buffer = []
        self._write_future = self._exec.submit(self._deferred_write, data)


    def background_sync_wide(self):
        if not self.cfg.enabled:
            return

        if self.is_writing():
            return

        data = self._wide_buffer
        self._wide_buffer = {}
        self._write_future = self._exec.submit(self._deferred_write_wide, data)


    def blocking_sync(self):
        if not self.cfg.enabled:
            return

        # wrap up in-progress sync
        if self._write_future is not None:
            self._write_future.result()

        if self.cfg.wide_format and self._wide_buffer:
            self.background_sync_wide()
            if self._write_future is not None:
                self._write_future.result()
        elif not self.cfg.wide_format and self._buffer:
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
                [point._asdict() for point in points]
            )
            connection.commit()


    def _deferred_write_wide(self, points: Dict[str, Dict[str, Any]]):
        if not points:
            return

        if not self.cfg.enabled:
            return
        assert self.engine is not None

        with TryConnectContextManager(self.engine) as connection:
            for timestamp, values in points.items():
                processed_values = {}
                column_types = {}

                for col, val in values.items():
                    if isinstance(val, dict) and "value" in val:
                        processed_values[col] = val["value"]
                        if "type" in val:
                            column_types[col] = val["type"]
                    else:
                        processed_values[col] = val
                        if isinstance(val, bool):
                            column_types[col] = "boolean"
                        elif isinstance(val, int):
                            column_types[col] = "integer"
                        elif isinstance(val, float):
                            column_types[col] = "float"
                        elif isinstance(val, str):
                            column_types[col] = "text"
                        else:
                            column_types[col] = "float"

                self._ensure_columns_exist(connection, list(values.keys()), column_types)

                columns = ", ".join([f'"{col}"' for col in processed_values.keys()])
                placeholders = ", ".join([f":{i}" for i in range(len(processed_values))])

                params = {"ts": timestamp}
                for i, (_col, val) in enumerate(processed_values.items()):
                    params[str(i)] = val

                query = text(f"""
                    INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
                    (time, {columns})
                    VALUES (TIMESTAMP :ts, {placeholders})
                    ON CONFLICT (time) DO UPDATE SET
                    {', '.join([f'"{col}" = EXCLUDED."{col}"' for col in processed_values.keys()])}
                """)

                connection.execute(query, params)
            connection.commit()


    def _ensure_columns_exist(
        self,
        connection: Connection,
        column_names: list[str],
        column_types: Dict[str, str] | None = None,
    ):
        for name in column_names:
            connection.execute(
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
                            ADD COLUMN "{name}" {column_types[name]};
                        END IF;
                    END $$;
                """)
            )
