from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Connection, Engine, TextClause, text

from lib_sql.connection import TryConnectContextManager
from lib_sql.inspection import get_all_columns, table_exists
from lib_sql.utils import SQLColumn, sanitize_keys
from lib_sql.writers.sql_writer import SqlWriter

logger = logging.getLogger(__name__)

TableCreationFactory = Callable[[str, str, list[SQLColumn]], TextClause]


class DynamicSchemaSqlWriter(SqlWriter):
    """Converts dictionary rows to database tables with dynamic schema evolution.
    Each dict key becomes a column, with automatic table and column creation.
    """

    def __init__(
        self,
        engine: Engine,
        table_name: str,
        table_creation_factory: TableCreationFactory,
        schema: str | None = None,
        default_column_type: str = "FLOAT",
    ):
        self.engine = engine
        self.table_name = table_name
        self.schema = schema
        self.default_column_type = default_column_type
        self.table_creation_factory = table_creation_factory

        self._table_created = False
        self._columns_initialized = False
        self._known_columns: set[str] = set()

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
            dict_points = sanitize_keys(rows)
            if not dict_points:
                return

            points_columns = self._extract_sorted_columns(dict_points)
            self._ensure_table_exists(connection, points_columns)
            self._ensure_known_columns_initialized(connection)
            self._add_columns(points_columns, connection)

            column_names = sorted(col.name for col in points_columns)
            ordered_dict_points = [
                {col_name: point.get(col_name) for col_name in column_names}
                for point in dict_points
            ]

            connection.execute(
                self._insert_sql(points_columns),
                ordered_dict_points,
            )
            connection.commit()

    def write(self, row: dict[str, Any]):
        self.write_many([row])

    def close(self):
        self.engine.dispose()

    # ========================================
    # Table & Schema Management
    # ========================================

    def _ensure_table_exists(
        self,
        connection: Connection,
        columns: list[SQLColumn],
    ):
        if self._table_created:
            return

        if table_exists(self.engine, table_name=self.table_name, schema=self.schema):
            self._table_created = True
            return


        table_sql = self.table_creation_factory(
            self.schema or "public",
            self.table_name,
            columns,
        )
        connection.execute(table_sql)
        connection.commit()
        self._table_created = True

    def _ensure_known_columns_initialized(self, connection: Connection):
        if self._columns_initialized:
            return

        columns = get_all_columns(self.engine, self.table_name, schema=self.schema)
        self._known_columns = {col["name"] for col in columns}
        self._columns_initialized = True

    def _add_columns(
        self,
        points_columns: list[SQLColumn],
        connection: Connection,
    ):
        if not points_columns:
            return

        new_columns = [
            col for col in points_columns
            if col.name not in self._known_columns
        ]

        for column in new_columns:
            connection.execute(text(f"""
                ALTER TABLE {self._table_ref}
                ADD COLUMN IF NOT EXISTS "{column.name}" {column.type}
            """))
            self._known_columns.add(column.name)

    # ========================================
    # SQL Query Generation
    # ========================================

    def _insert_sql(self, columns: list[SQLColumn]):
        column_names = sorted(col.name for col in columns)
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

    def _extract_sorted_columns(self, dict_points: list[dict[str, Any]]) -> list[SQLColumn]:
        """Extract and sort columns from data points, returning SQLColumn objects with inferred types."""
        points_columns = set[str]()
        for point in dict_points:
            points_columns.update(point.keys())

        column_types = self._infer_column_types(dict_points)
        return [
            SQLColumn(name=col_name, type=column_types.get(col_name, self.default_column_type))
            for col_name in sorted(points_columns)
        ]

    def _infer_column_types(self, dict_points: list[dict[str, Any]]):
        column_types: dict[str, str] = {}

        for point in dict_points:
            for col_name, value in point.items():
                if col_name in column_types:
                    continue

                if value is None:
                    continue

                if isinstance(value, bool):
                    column_types[col_name] = "BOOLEAN"
                elif isinstance(value, int):
                    column_types[col_name] = "INTEGER"
                elif isinstance(value, float):
                    column_types[col_name] = "DOUBLE PRECISION"
                elif isinstance(value, str):
                    column_types[col_name] = "TEXT"
                elif isinstance(value, datetime):
                    column_types[col_name] = "TIMESTAMP WITH TIME ZONE"
                else:
                    column_types[col_name] = self.default_column_type

        return column_types
