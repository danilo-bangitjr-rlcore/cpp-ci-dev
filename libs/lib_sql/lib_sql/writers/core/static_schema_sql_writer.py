from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

from sqlalchemy import Connection, Engine, text

from lib_sql.connection import TryConnectContextManager
from lib_sql.inspection import table_exists
from lib_sql.utils import SQLColumn, sanitize_keys
from lib_sql.writers.sql_writer import SqlWriter

logger = logging.getLogger(__name__)

# Type alias for table creation factory function
TableCreationFactory = Callable[[str, str, list[SQLColumn]], Any]


class StaticSchemaSqlWriter(SqlWriter):
    """Converts dictionary rows to database tables with a predefined, static schema.
    The schema is fixed at initialization time and no new columns can be added.
    """

    def __init__(
        self,
        engine: Engine,
        table_name: str,
        columns: list[SQLColumn],
        table_creation_factory: TableCreationFactory,
        schema: str | None = None,
    ):
        self.engine = engine
        self.table_name = table_name
        self.schema = schema
        self.columns = columns
        self.table_creation_factory = table_creation_factory

        self._table_created = False
        self._column_names = {col.name for col in columns}
        self._primary_columns = [col.name for col in columns if col.primary]

    @property
    def _table_ref(self):
        return f"{self.schema}.{self.table_name}" if self.schema else self.table_name

    # ========================================
    # Core Public API
    # ========================================

    def write_many(self, rows: Sequence[dict[str, Any]]):
        if not rows:
            return

        with TryConnectContextManager(self.engine) as connection:
            self._ensure_table_exists(connection)

            dict_points = self._prepare_data_for_insert(rows)
            connection.execute(
                self._insert_sql,
                dict_points,
            )
            connection.commit()

    def write(self, row: dict[str, Any]):
        self.write_many([row])

    def flush(self):
        """No-op flush for StaticSchemaSqlWriter."""

    def close(self):
        self.engine.dispose()

    # ========================================
    # Table & Schema Management
    # ========================================

    def _ensure_table_exists(self, connection: Connection):
        if self._table_created:
            return

        if table_exists(self.engine, table_name=self.table_name, schema=self.schema):
            self._table_created = True
            return

        # Use the provided factory function
        table_sql = self.table_creation_factory(
            self.schema or "public",
            self.table_name,
            self.columns,
        )
        connection.execute(table_sql)

        # Ensure table creation is committed and visible
        connection.commit()
        self._table_created = True

    # ========================================
    # SQL Query Generation
    # ========================================

    @cached_property
    def _insert_sql(self):
        """Pre-computed SQL for inserting data with the static schema columns."""
        if not self.columns:
            # Handle edge case of empty schema - insert into default id column
            return text(f"INSERT INTO {self._table_ref} DEFAULT VALUES")

        column_names = sorted(col.name for col in self.columns)
        columns_list = ", ".join(f'"{col}"' for col in column_names)
        placeholders = ", ".join(f":{col}" for col in column_names)

        return text(f"""
            INSERT INTO {self._table_ref}
            ({columns_list})
            VALUES ({placeholders})
        """)

    # ========================================
    # Data Processing & Transformation
    # ========================================

    def _prepare_data_for_insert(self, rows: Sequence[dict[str, Any]]):
        dict_points = sanitize_keys(rows)

        # Validate that all columns in the data match our schema
        for i, point in enumerate(dict_points):
            unexpected_columns = set(point.keys()) - self._column_names
            if unexpected_columns:
                raise ValueError(
                    f"Row {i} contains unexpected columns not in schema: {sorted(unexpected_columns)}. "
                    f"Schema columns: {sorted(self._column_names)}",
                )

        return [
            {col.name: point.get(col.name) for col in self.columns}
            for point in dict_points
        ]
