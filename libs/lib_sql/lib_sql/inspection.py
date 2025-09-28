import logging
from typing import Literal, NamedTuple

from lib_utils.list import find
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class ColumnInfo(NamedTuple):
    column_name: str
    data_type: str
    is_nullable: Literal["YES", "NO"]


def table_exists(engine: Engine, table_name: str, schema: str | None = None) -> bool:
    iengine = inspect(engine)
    existing_tables = iengine.get_table_names(schema)
    return table_name in existing_tables


def column_exists(
    engine: Engine,
    table_name: str,
    column_name: str,
    schema: str | None = None,
) -> bool:
    try:
        iengine = inspect(engine)
        columns = iengine.get_columns(table_name, schema=schema)
        return any(col["name"] == column_name for col in columns)
    except SQLAlchemyError:
        return False


def get_column_type(
    engine: Engine,
    table_name: str,
    column_name: str,
    schema: str | None = None,
):
    iengine = inspect(engine)
    columns = iengine.get_columns(table_name, schema=schema)
    column = find(lambda col: col["name"] == column_name, columns)
    assert column is not None, "SQL Error, column not found"
    return column["type"]


def get_all_columns(engine: Engine, table_name: str, schema: str | None = None):
    iengine = inspect(engine)
    return iengine.get_columns(table_name, schema=schema)


def table_count(engine: Engine, table_name: str, schema: str | None = None) -> int:
    table_ref = f"{schema}.{table_name}" if schema else table_name

    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_ref}")).scalar()
        return int(result) if result is not None else 0


def get_table_schema(engine: Engine, table_name: str, schema: str | None = None) -> list[ColumnInfo]:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table_name
                ORDER BY ordinal_position
            """),
            {"schema": schema or "public", "table_name": table_name},
        ).fetchall()

        return [
            ColumnInfo(
                column_name=row.column_name,
                data_type=row.data_type,
                is_nullable=row.is_nullable,
            )
            for row in result
        ]
