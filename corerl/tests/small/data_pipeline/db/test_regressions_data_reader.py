from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, text

from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig
from corerl.data_pipeline.db.data_writer import DataWriter

# Regression test for schema-qualified table replacement in DataReader.query
# Ensures that :table is replaced with schema.table when schema != public

@pytest.fixture()
def non_public_db_config(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    port = tsdb_engine.url.port
    assert port is not None
    return TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=port,
        db_name=tsdb_tmp_db_name,
        table_name="tags",
        table_schema="solar",
    )

@pytest.fixture()
def non_public_schema_initialized(tsdb_engine: Engine):
    with tsdb_engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS solar"))
        conn.commit()
    return True

@pytest.fixture()
def reader_writer_non_public(
    non_public_db_config: TagDBConfig,
    non_public_schema_initialized: bool,
):
    reader = DataReader(db_cfg=non_public_db_config)
    writer = DataWriter(cfg=non_public_db_config)
    yield reader, writer
    reader.close()
    writer.close()


def test_query_schema_qualified(reader_writer_non_public: tuple[DataReader, DataWriter]):
    """
    Test that DataReader.query properly replaces :table placeholder with schema-qualified table name.

    When the table schema is non-public (e.g., 'solar'), the query should use 'solar.tags'
    instead of just 'tags' to ensure it works correctly across different database schemas.
    """
    reader, writer = reader_writer_non_public

    # Ensure table exists by writing a dummy data point first
    ts = datetime.now(tz=UTC)
    writer.write(timestamp=ts, name="test", val=1.0)
    writer.blocking_sync()

    df = reader.query("SELECT time FROM :table LIMIT 1")
    assert "time" in df.columns or df.empty


def test_writer_and_reader_round_trip(reader_writer_non_public: tuple[DataReader, DataWriter]):
    """
    Test end-to-end write and read operations with schema-qualified table names.

    Verifies that data written to a non-public schema table can be successfully
    read back using DataReader.query with the :table placeholder replacement.
    """
    reader, writer = reader_writer_non_public

    ts = datetime.now(tz=UTC)
    writer.write(timestamp=ts, name="foo", val=1.23)
    writer.blocking_sync()

    df = reader.query("SELECT time, name FROM :table WHERE name='foo'")
    assert not df.empty
    assert (df["name"] == "foo").any()
