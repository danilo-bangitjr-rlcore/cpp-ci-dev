from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from lib_sql.inspection import table_exists
from lib_sql.utils import SQLColumn, create_tsdb_table_query
from lib_sql.writers.static_schema_sql_writer import StaticSchemaSqlWriter

pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.docker",
    "test.infrastructure.utils.tsdb",
]


@pytest.fixture
def test_schema():
    """Create a test schema with various column types."""
    return [
        SQLColumn(name="time", type="TIMESTAMPTZ", nullable=False),
        SQLColumn(name="metric_a", type="DOUBLE PRECISION", nullable=False),
        SQLColumn(name="metric_b", type="INTEGER", nullable=True),
        SQLColumn(name="category", type="TEXT", nullable=False),
        SQLColumn(name="is_active", type="BOOLEAN", nullable=True),
    ]


@pytest.fixture
def test_writer(tsdb_engine: Engine, test_schema: list[SQLColumn]):
    """Create a test writer instance with predefined schema."""
    return StaticSchemaSqlWriter(
        engine=tsdb_engine,
        table_name="test_static_table",
        columns=test_schema,
        table_creation_factory=lambda schema, table, columns: create_tsdb_table_query(
            schema=schema,
            table=table,
            columns=columns,
        ),
        schema="public",
    )


def test_basic_write_operations(test_writer: StaticSchemaSqlWriter, tsdb_engine: Engine):
    """Test basic write operations with static schema."""
    # Test empty write (should be no-op)
    test_writer.write_many([])
    assert not table_exists(tsdb_engine, "test_static_table", schema="public")

    # Test single write with valid data
    single_point = {
        "time": "2024-01-01 12:00:00+00",
        "metric_a": 1.5,
        "metric_b": 42,
        "category": "test",
        "is_active": True,
    }
    test_writer.write(single_point)
    assert table_exists(tsdb_engine, "test_static_table", schema="public")
    with tsdb_engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM public.test_static_table")).fetchall()
        assert len(result) == 1
        row = result[0]
        assert row.metric_a == 1.5
        assert row.metric_b == 42
        assert row.category == "test"
        assert row.is_active is True

    # Test batch write
    batch_data = [
        {
            "time": "2024-01-01 13:00:00+00",
            "metric_a": 2.5,
            "metric_b": 100,
            "category": "batch1",
            "is_active": False,
        },
        {
            "time": "2024-01-01 14:00:00+00",
            "metric_a": 3.5,
            "category": "batch2",
        },
    ]
    test_writer.write_many(batch_data)
    with tsdb_engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM public.test_static_table ORDER BY time")).fetchall()
        assert len(result) == 3

        # Second row
        assert result[1].metric_a == 2.5
        assert result[1].metric_b == 100
        assert result[1].category == "batch1"
        assert result[1].is_active is False

        # Third row (with nullable fields)
        assert result[2].metric_a == 3.5
        assert result[2].metric_b is None
        assert result[2].category == "batch2"
        assert result[2].is_active is None

    test_writer.close()


def test_schema_enforcement(test_schema: list[SQLColumn], tsdb_engine: Engine):
    """Test that the writer raises an exception for columns not defined in the schema."""
    # Test invalid data with extra columns
    invalid_writer = StaticSchemaSqlWriter(
        engine=tsdb_engine,
        table_name="test_static_table_invalid",
        columns=test_schema,
        table_creation_factory=lambda schema, table, columns: create_tsdb_table_query(
            schema=schema,
            table=table,
            columns=columns,
            partition_column=None,
            index_columns=[],
        ),
        schema="public",
    )

    data_with_extra_columns = [
        {
            "time": "2024-01-01 15:00:00+00",
            "metric_a": 1.0,
            "metric_b": 10,
            "category": "valid",
            "is_active": True,
            "extra_column": "should_cause_error",  # Not in schema
            "another_extra": 999,  # Not in schema
        },
    ]
    with pytest.raises(ValueError, match=r"Row 0 contains unexpected columns not in schema"):
        invalid_writer.write_many(data_with_extra_columns)


def test_table_creation_with_schema(test_schema: list[SQLColumn], tsdb_engine: Engine):
    """Test that table is created with correct schema including constraints."""
    writer = StaticSchemaSqlWriter(
        engine=tsdb_engine,
        table_name="test_schema_creation",
        columns=test_schema,
        table_creation_factory=lambda schema, table, columns: create_tsdb_table_query(
            schema=schema,
            table=table,
            columns=columns,
            partition_column=None,
            index_columns=[],
        ),
        schema="public",
    )

    writer.write({"time": "2024-01-01 17:00:00+00", "metric_a": 1.0, "category": "test"})
    with tsdb_engine.connect() as conn:
        columns_query = text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'test_schema_creation'
            ORDER BY column_name
        """)
        columns = conn.execute(columns_query).fetchall()

        column_info = {row.column_name: (row.data_type, row.is_nullable) for row in columns}

        # Verify expected columns exist with correct types and constraints
        assert column_info["metric_a"][0] == "double precision"
        assert column_info["metric_a"][1] == "NO"  # NOT NULL

        assert column_info["metric_b"][0] == "integer"
        assert column_info["metric_b"][1] == "YES"  # NULLABLE

        assert column_info["category"][0] == "text"
        assert column_info["category"][1] == "NO"  # NOT NULL

        assert column_info["is_active"][0] == "boolean"
        assert column_info["is_active"][1] == "YES"  # NULLABLE

        assert column_info["time"][0] == "timestamp with time zone"
        assert column_info["time"][1] == "NO"  # Primary key is NOT NULL
