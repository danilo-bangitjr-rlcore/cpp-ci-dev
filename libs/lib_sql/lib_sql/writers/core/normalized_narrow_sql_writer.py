from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from functools import cached_property
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    Connection,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Index,
    MetaData,
    Table,
    Text,
    bindparam,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import REGCLASS
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import SQLAlchemyError

from lib_sql.connection import TryConnectContextManager
from lib_sql.inspection import table_exists
from lib_sql.utils import sanitize_keys
from lib_sql.writers.sql_writer import SqlWriter

logger = logging.getLogger(__name__)


class NormalizedNarrowSqlWriter(SqlWriter):
    """A normalized narrow writer that creates two tables:

    1. metric_lookup: Interns metric names into BIGSERIAL PRIMARY KEY IDs
    2. data_table: Stores (timestamp, metric_id, value) in narrow format

    Uses an in-memory cache for metric name -> ID mapping to minimize lookup table queries.

    Note: This writer is designed for PostgreSQL and uses PostgreSQL-specific features
    like ON CONFLICT DO NOTHING. It will not work with SQLite or other databases.
    """

    def __init__(
        self,
        engine: Engine,
        data_table_name: str,
        lookup_table_name: str | None = None,
        schema: str | None = None,
        *,
        time_column: str = "timestamp",
        chunk_time_interval: str = "1d",
    ):
        self.engine = engine
        self.data_table_name = data_table_name
        self.lookup_table_name = lookup_table_name or f"{data_table_name}_metadata"
        self.schema = schema
        self.time_column = time_column
        self.chunk_time_interval = chunk_time_interval

        self._tables_created = False
        self._tsdb_configured = False
        self._metric_cache: dict[str, int] = {}
        self._cache_lock = threading.Lock()
        self._ddl_lock = threading.Lock()

        self.metadata = MetaData(schema=schema)

        self.lookup_table = Table(
            self.lookup_table_name,
            self.metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("metric_name", Text, nullable=False, unique=True),
        )

        self.data_table = Table(
            self.data_table_name,
            self.metadata,
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("metric_id", BigInteger, ForeignKey(self.lookup_table.c.id), nullable=False),
            Column("value", Float, nullable=False),
            Index(f"{self.data_table_name}_metric_id_idx", "metric_id"),
        )

        if self.time_column not in self.data_table.c:
            raise ValueError(f"Time column '{self.time_column}' is not defined on the data table")

    @property
    def _data_table_ref(self):
        return f"{self.schema}.{self.data_table_name}" if self.schema else self.data_table_name

    @property
    def _lookup_table_ref(self):
        return f"{self.schema}.{self.lookup_table_name}" if self.schema else self.lookup_table_name

    @property
    def _qualified_data_table(self):
        # NOTE: quoted identifiers for hypertable creation
        return f'"{self.schema}"."{self.data_table_name}"' if self.schema else f'"{self.data_table_name}"'

    # ========================================
    # Core Public API
    # ========================================

    def write_many(self, rows: Sequence[dict[str, Any]]):
        """Write multiple rows in narrow format: {'timestamp': ..., 'metric': ..., 'value': ...}"""
        if not rows:
            return

        with TryConnectContextManager(self.engine) as connection:
            self._ensure_tables_exist(connection)

            data_rows = []
            for row in rows:
                sanitized_row = sanitize_keys([row])[0]

                if 'timestamp' not in sanitized_row or 'metric' not in sanitized_row or 'value' not in sanitized_row:
                    raise ValueError(
                        f"Row must contain 'timestamp', 'metric', and 'value' fields. "
                        f"Got: {list(sanitized_row.keys())}",
                    )

                metric_name = sanitized_row['metric']
                metric_id = self._get_or_create_metric_id(connection, metric_name)

                data_rows.append({
                    'timestamp': sanitized_row['timestamp'],
                    'metric_id': metric_id,
                    'value': sanitized_row['value'],
                })

            if data_rows:
                connection.execute(self._data_insert_stmt, data_rows)
            connection.commit()

    def write(self, row: dict[str, Any]):
        self.write_many([row])

    def flush(self):
        """No-op flush for NormalizedNarrowSqlWriter."""

    def close(self):
        self.engine.dispose()

    # ========================================
    # Table & Schema Management
    # ========================================

    def _ensure_tables_exist(self, connection: Connection):
        if self._tables_created and self._tsdb_configured:
            return

        with self._ddl_lock:
            if self._tables_created and self._tsdb_configured:
                return

            lookup_exists = table_exists(self.engine, table_name=self.lookup_table_name, schema=self.schema)
            data_exists = table_exists(self.engine, table_name=self.data_table_name, schema=self.schema)

            if not (lookup_exists and data_exists):
                self.metadata.create_all(self.engine, checkfirst=True)

            self._tables_created = True

            if not self._tsdb_configured:
                self._configure_tsdb_features(connection)

    def _configure_tsdb_features(self, connection: Connection) -> None:
        """Configure TimescaleDB hypertable and compression for the data table."""
        try:
            self._create_hypertable(connection)
            self._enable_compression(connection)
            connection.commit()
            self._tsdb_configured = True
        except SQLAlchemyError as exc:
            connection.rollback()
            schema = self.schema or "public"
            logger.exception(
                "Failed to configure TimescaleDB features for hypertable '%s' in schema '%s'",
                self.data_table_name,
                schema,
            )
            raise exc

    def _create_hypertable(self, connection: Connection) -> None:
        """Create TimescaleDB hypertable."""
        stmt = text(
            """
            SELECT create_hypertable(
                :table_name,
                :time_column,
                if_not_exists => TRUE,
                chunk_time_interval => CAST(:chunk_interval AS INTERVAL)
            );
            """,
        ).bindparams(bindparam("table_name", type_=REGCLASS))

        connection.execute(
            stmt,
            {
                "table_name": self._data_table_ref,
                "time_column": self.time_column,
                "chunk_interval": self.chunk_time_interval,
            },
        )

    def _enable_compression(self, connection: Connection) -> None:
        """Enable TimescaleDB compression with metric_id segmentation."""
        stmt = text(
            f"""
            ALTER TABLE {self._qualified_data_table} SET (
                timescaledb.compress = TRUE,
                timescaledb.compress_segmentby = :segmentby
            );
            """,
        )
        connection.execute(stmt, {"segmentby": "metric_id"})

    # ========================================
    # SQL Query Generation
    # ========================================

    @cached_property
    def _data_insert_stmt(self):
        return insert(self.data_table)

    @cached_property
    def _lookup_insert_stmt(self):
        stmt = postgresql_insert(self.lookup_table)
        return stmt.on_conflict_do_nothing(index_elements=['metric_name']).returning(self.lookup_table.c.id)

    def _create_lookup_select_stmt(self, metric_name: str):
        return select(self.lookup_table.c.id).where(self.lookup_table.c.metric_name == metric_name)

    # ========================================
    # Metric ID Management & Caching
    # ========================================

    def _get_or_create_metric_id(self, connection: Connection, metric_name: str) -> int:
        """Get metric ID from cache or database, creating if necessary."""
        with self._cache_lock:
            cached_id = self._metric_cache.get(metric_name)
            if cached_id is not None:
                return cached_id

        with self._cache_lock:
            cached_id = self._metric_cache.get(metric_name)
            if cached_id is not None:
                return cached_id

            insert_stmt = self._lookup_insert_stmt.values(metric_name=metric_name)
            result = connection.execute(insert_stmt)
            inserted_id = result.fetchone()

            if inserted_id:
                metric_id = inserted_id[0]
            else:
                select_stmt = self._create_lookup_select_stmt(metric_name)
                result = connection.execute(select_stmt)
                row = result.fetchone()
                if not row:
                    raise RuntimeError(f"Failed to find or create metric ID for '{metric_name}'")
                metric_id = row[0]

            self._metric_cache[metric_name] = metric_id
            return metric_id
