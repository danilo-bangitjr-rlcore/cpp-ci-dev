from pathlib import Path

import pytest
from sqlalchemy import URL, Column, Engine, Integer, MetaData, String, Table, create_engine

from lib_sql.inspection import column_exists, get_all_columns, get_column_type, table_exists


@pytest.fixture
def test_engine(tmp_path: Path):
    """Fixture for a test SQLite engine with a table."""
    db_path = tmp_path / "test.db"
    url = URL.create(drivername="sqlite", database=str(db_path))
    engine = create_engine(url)
    metadata = MetaData()
    Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(50)),
    )
    metadata.create_all(engine)
    yield engine
    # Cleanup
    metadata.drop_all(engine)


def test_table_exists_true(test_engine: Engine):
    """
    Test table exists when it does.
    """
    assert table_exists(test_engine, "test_table", schema=None)


def test_table_exists_false(test_engine: Engine):
    """
    Test table exists when it doesn't.
    """
    assert not table_exists(test_engine, "nonexistent_table", schema=None)


def test_column_exists_true(test_engine: Engine):
    """
    Test column exists when it does.
    """
    assert column_exists(test_engine, "test_table", "id", schema=None)


def test_column_exists_false(test_engine: Engine):
    """
    Test column exists when it doesn't.
    """
    assert not column_exists(test_engine, "test_table", "nonexistent_column", schema=None)


def test_column_exists_table_not_exists(test_engine: Engine):
    """
    Test column exists when table doesn't exist.
    """
    assert not column_exists(test_engine, "nonexistent_table", "id", schema=None)


def test_get_column_type(test_engine: Engine):
    """
    Test get column type.
    """
    col_type = get_column_type(test_engine, "test_table", "id", schema=None)
    assert str(col_type) == "INTEGER"


def test_get_all_columns(test_engine: Engine):
    """
    Test get all columns.
    """
    columns = get_all_columns(test_engine, "test_table", schema=None)
    assert len(columns) == 2
    assert columns[0]["name"] == "id"
    assert columns[1]["name"] == "name"
