import logging

from lib_utils.list import find
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

def table_exists(engine: Engine, table_name: str, schema: str | None = None) -> bool:
    iengine = inspect(engine)
    existing_tables = iengine.get_table_names(schema)
    return table_name in existing_tables


def column_exists(
    engine: Engine, table_name: str, column_name: str, schema: str | None = None,
) -> bool:
    try:
        iengine = inspect(engine)
        columns = iengine.get_columns(table_name, schema=schema)
        return any(col["name"] == column_name for col in columns)
    except SQLAlchemyError:
        return False


def get_column_type(
    engine: Engine, table_name: str, column_name: str, schema: str | None = None,
):
    assert column_exists(
        engine, table_name, column_name, schema,
    ), "SQL Error, column not found"
    iengine = inspect(engine)
    columns = iengine.get_columns(table_name, schema=schema)
    column = find(lambda col: col["name"] == column_name, columns)
    assert column is not None  # Check for pyright, since we already checked that column exists
    return column["type"]


def get_all_columns(engine: Engine, table_name: str, schema: str | None = None):
    iengine = inspect(engine)
    return iengine.get_columns(table_name, schema=schema)
