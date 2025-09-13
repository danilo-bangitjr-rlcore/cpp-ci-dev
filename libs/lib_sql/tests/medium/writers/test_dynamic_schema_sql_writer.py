from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from lib_sql.inspection import table_exists
from lib_sql.utils import create_tsdb_table_query
from lib_sql.writers.dynamic_schema_sql_writer import DynamicSchemaSqlWriter

pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.docker",
    "test.infrastructure.utils.tsdb",
]


@pytest.fixture
def test_writer(tsdb_engine: Engine) -> DynamicSchemaSqlWriter:
    """Create a test writer instance."""
    return DynamicSchemaSqlWriter(
        engine=tsdb_engine,
        table_name="test_table",
        table_creation_factory=lambda schema, table, columns: create_tsdb_table_query(
            schema=schema,
            table=table,
            columns=columns,
            partition_column=None,
            index_columns=[],
        ),
        schema="public",
    )


def test_basic_write_operations(test_writer: DynamicSchemaSqlWriter, tsdb_engine: Engine) -> None:
    """Test basic write operations: single write, batch write, empty write, and close."""
    test_writer.write_many([])
    assert not table_exists(tsdb_engine, "test_table", schema="public")

    single_point = {
        "metric_a": 1.5,
        "metric_b": 42,
        "category": "test",
        "time": datetime.now(),
    }
    test_writer.write(single_point)
    with tsdb_engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM public.test_table")).fetchall()
        assert len(result) == 1
        row = result[0]
        assert row.metric_a == 1.5
        assert row.metric_b == 42
        assert row.category == "test"
        assert row.time is not None

    test_writer.close()
