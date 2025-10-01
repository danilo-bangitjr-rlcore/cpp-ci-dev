from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError

from lib_sql.inspection import get_table_schema, table_count, table_exists
from lib_sql.writers.core.normalized_narrow_sql_writer import NormalizedNarrowSqlWriter

pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.docker",
    "test.infrastructure.utils.tsdb",
]


# ============================================================================
# Fixtures & Utilities
# ============================================================================


@pytest.fixture
def test_writer(tsdb_engine: Engine):
    writer = NormalizedNarrowSqlWriter(
        engine=tsdb_engine,
        data_table_name="test_narrow_data",
        lookup_table_name="test_narrow_metadata",
        schema="public",
    )
    yield writer

    # Cleanup
    try:
        writer.close()
    except Exception:
        pass


@pytest.fixture
def writer_factory(tsdb_engine: Engine):
    """Factory for creating writers with unique table names."""
    created_writers = []

    def _create_writer(base_name: str = "test", schema: str = "public"):
        data_table = f"{base_name}_data_{len(created_writers)}"
        lookup_table = f"{base_name}_metadata_{len(created_writers)}"
        writer = NormalizedNarrowSqlWriter(
            engine=tsdb_engine,
            data_table_name=data_table,
            lookup_table_name=lookup_table,
            schema=schema,
        )
        created_writers.append(writer)
        return writer

    yield _create_writer

    # Cleanup
    for writer in created_writers:
        try:
            writer.close()
        except Exception:
            pass


@pytest.fixture
def sample_metric_data():
    """Reusable metric data for tests."""
    return [
        {"timestamp": "2024-01-01 12:00:00+00", "metric": "accuracy", "value": 0.95},
        {"timestamp": "2024-01-01 13:00:00+00", "metric": "loss", "value": 0.1},
        {"timestamp": "2024-01-01 14:00:00+00", "metric": "f1_score", "value": 0.87},
    ]


@pytest.fixture
def special_char_metrics():
    """Test data with special characters in metric names."""
    return [
        {"timestamp": "2024-01-01 12:00:00+00", "metric": "metric with spaces", "value": 1.0},
        {"timestamp": "2024-01-01 13:00:00+00", "metric": "metric-with-hyphens", "value": 2.0},
        {"timestamp": "2024-01-01 14:00:00+00", "metric": "metric_with_underscores", "value": 3.0},
        {"timestamp": "2024-01-01 15:00:00+00", "metric": "metric.with.dots", "value": 4.0},
        {"timestamp": "2024-01-01 16:00:00+00", "metric": "metric/with/slashes", "value": 5.0},
    ]


@pytest.fixture
def case_variant_metrics():
    """Test data with different case variants of the same metric."""
    return [
        {"timestamp": "2024-01-01 12:00:00+00", "metric": "accuracy", "value": 0.95},
        {"timestamp": "2024-01-01 13:00:00+00", "metric": "Accuracy", "value": 0.96},
        {"timestamp": "2024-01-01 14:00:00+00", "metric": "ACCURACY", "value": 0.97},
    ]


def assert_lookup_table_schema(engine: Engine, schema: str, table_name: str):
    """Assert lookup table has correct schema structure."""
    columns = get_table_schema(engine, table_name=table_name, schema=schema)

    assert len(columns) == 2, f"Expected 2 columns, got {len(columns)}"
    assert columns[0].column_name == "id"
    assert columns[0].data_type == "bigint"
    assert columns[0].is_nullable == "NO"
    assert columns[1].column_name == "metric_name"
    assert columns[1].data_type == "text"
    assert columns[1].is_nullable == "NO"


def assert_data_table_schema(engine: Engine, schema: str, table_name: str):
    """Assert data table has correct schema structure."""
    columns = get_table_schema(engine, table_name=table_name, schema=schema)

    assert len(columns) == 3, f"Expected 3 columns, got {len(columns)}"
    assert columns[0].column_name == "timestamp"
    assert columns[0].data_type == "timestamp with time zone"
    assert columns[1].column_name == "metric_id"
    assert columns[1].data_type == "bigint"
    assert columns[2].column_name == "value"
    assert columns[2].data_type == "double precision"


def get_written_metrics(engine: Engine, schema: str, data_table: str, lookup_table: str):
    """Get all written metrics with their values."""
    with engine.connect() as conn:
        return conn.execute(
            text(f"""
            SELECT d.timestamp, m.metric_name, d.value
            FROM {schema}.{data_table} d
            JOIN {schema}.{lookup_table} m ON d.metric_id = m.id
            ORDER BY d.timestamp
            """),
        ).fetchall()


def get_metric_lookup_data(engine: Engine, schema: str, table_name: str):
    """Get metric lookup data for verification."""
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT id, metric_name FROM {schema}.{table_name} ORDER BY id"),
        ).fetchall()


def assert_metric_cached(writer: NormalizedNarrowSqlWriter, metric_name: str, expected_id: int | None = None):
    """Assert that a metric is cached with optional ID verification."""
    assert metric_name in writer._metric_cache, f"Metric '{metric_name}' not found in cache"
    cached_id = writer._metric_cache[metric_name]
    assert isinstance(cached_id, int), f"Cached ID should be int, got {type(cached_id)}"
    assert cached_id > 0, f"Cached ID should be positive, got {cached_id}"

    if expected_id is not None:
        assert cached_id == expected_id, f"Expected ID {expected_id}, got {cached_id}"


def assert_cache_contains_metrics(writer: NormalizedNarrowSqlWriter, expected_metrics: set[str]):
    """Assert that cache contains all expected metrics."""
    cached_metrics = set(writer._metric_cache.keys())
    assert expected_metrics.issubset(cached_metrics), (
        f"Expected metrics {expected_metrics} not all found in cache {cached_metrics}"
    )


def assert_unique_cached_ids(writer: NormalizedNarrowSqlWriter):
    """Assert that all cached metric IDs are unique."""
    cached_ids = list(writer._metric_cache.values())
    unique_ids = set(cached_ids)
    assert len(cached_ids) == len(unique_ids), f"Duplicate IDs found in cache: {cached_ids}"


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_basic_write_operations(
    test_writer: NormalizedNarrowSqlWriter,
    tsdb_engine: Engine,
    sample_metric_data: list[dict],
):
    """
    Test basic write operations with normalized narrow format.

    Validates that single and batch writes correctly normalize metric names
    into a lookup table and store data in narrow format. This ensures the
    core functionality of metric name deduplication works as expected.
    """
    # Test single write operation
    single_point = sample_metric_data[0]  # Use first item from fixture
    test_writer.write(single_point)

    # Verify single write results using utility function
    written_data = get_written_metrics(
        tsdb_engine, "public", "test_narrow_data", "test_narrow_metadata",
    )
    assert len(written_data) == 1
    row = written_data[0]
    assert row.metric_name == single_point["metric"]
    assert row.value == single_point["value"]

    # Verify metric is cached
    assert_metric_cached(test_writer, single_point["metric"])

    # Test batch write operation with remaining data
    batch_data = sample_metric_data[1:]  # Use remaining items from fixture
    test_writer.write_many(batch_data)

    # Verify all data using utility function
    all_written_data = get_written_metrics(
        tsdb_engine, "public", "test_narrow_data", "test_narrow_metadata",
    )
    assert len(all_written_data) == len(sample_metric_data)

    # Verify all expected metrics are present and cached
    expected_metrics = {item["metric"] for item in sample_metric_data}
    assert_cache_contains_metrics(test_writer, expected_metrics)

    # Verify data integrity by checking each expected metric/value pair
    written_metrics = {(row.metric_name, row.value) for row in all_written_data}
    expected_pairs = {(item["metric"], item["value"]) for item in sample_metric_data}
    assert written_metrics == expected_pairs, f"Data mismatch: expected {expected_pairs}, got {written_metrics}"


def test_tsdb_hypertable_and_compression_configuration(
    writer_factory: Callable[..., NormalizedNarrowSqlWriter],
    tsdb_engine: Engine,
    sample_metric_data: list[dict],
):
    writer = writer_factory(base_name="tsdb_config")
    writer.write_many(sample_metric_data)

    schema = writer.schema or "public"
    data_table = writer.data_table_name

    with tsdb_engine.connect() as conn:
        try:
            hypertable_row = conn.execute(
                text(
                    """
                        SELECT compression_enabled, chunk_interval
                        FROM timescaledb_information.hypertables
                        WHERE hypertable_schema = :schema
                          AND hypertable_name = :table
                        """,
                ),
                {"schema": schema, "table": data_table},
            ).one()

            compression_enabled = hypertable_row.compression_enabled
            chunk_interval = hypertable_row.chunk_interval
        except ProgrammingError:
            conn.rollback()

            compression_enabled = conn.execute(
                text(
                    """
                        SELECT compression_enabled
                        FROM timescaledb_information.hypertables
                        WHERE hypertable_schema = :schema
                          AND hypertable_name = :table
                        """,
                ),
                {"schema": schema, "table": data_table},
            ).scalar_one()

            chunk_interval = conn.execute(
                text(
                    """
                        SELECT interval '1 microsecond' * d.interval_length AS chunk_interval
                        FROM _timescaledb_catalog.dimension d
                        JOIN _timescaledb_catalog.hypertable h
                          ON h.id = d.hypertable_id
                        WHERE h.schema_name = :schema
                          AND h.table_name = :table
                          AND d.column_name = :time_column
                        LIMIT 1
                        """,
                ),
                {
                    "schema": schema,
                    "table": data_table,
                    "time_column": writer.time_column,
                },
            ).scalar_one()

        assert compression_enabled is True
        assert chunk_interval == timedelta(days=1)

        matching_rows: list[dict[str, Any]] = []
        schema_candidates = ("hypertable_schema", "schema_name", "schema")
        table_candidates = ("hypertable_name", "table_name", "relation_name", "name")

        for view_name in (
            "timescaledb_information.hypertable_compression_settings",
            "timescaledb_information.compression_settings",
        ):
            try:
                compression_rows = conn.execute(
                    text(f"SELECT * FROM {view_name}"),
                ).mappings().all()
            except ProgrammingError:
                conn.rollback()
                continue

            for row in compression_rows:
                schema_value = next((row.get(candidate) for candidate in schema_candidates if candidate in row), None)
                table_value = next((row.get(candidate) for candidate in table_candidates if candidate in row), None)
                if schema_value == schema and table_value == data_table:
                    matching_rows.append(dict(row))

            if matching_rows:
                break

        assert matching_rows, "Compression settings not found for hypertable"

        segmentby_value: str | None = None
        for row in matching_rows:
            if row.get("segmentby_column"):
                segmentby_value = row["segmentby_column"]
                break
            if row.get("segmentby_column_name"):
                segmentby_value = row["segmentby_column_name"]
                break
            if row.get("segmentby"):
                segmentby_value = row["segmentby"]
                break
            if row.get("segmentby_column_index") is not None:
                segmentby_value = row.get("attname") or row.get("column_name")
                if segmentby_value:
                    break

        assert segmentby_value == "metric_id"


def test_schema_validation(tsdb_engine: Engine):
    """
    Test that the writer validates required fields.

    Ensures the writer properly rejects incomplete data by validating
    that all required fields (timestamp, metric, value) are present.
    This prevents silent data corruption from missing fields.
    """
    writer = NormalizedNarrowSqlWriter(
        engine=tsdb_engine,
        data_table_name="test_validation",
        schema="public",
    )

    with pytest.raises(ValueError, match=r"Row must contain 'timestamp', 'metric', and 'value' fields"):
        writer.write({"timestamp": "2024-01-01 12:00:00+00", "value": 1.0})

    with pytest.raises(ValueError, match=r"Row must contain 'timestamp', 'metric', and 'value' fields"):
        writer.write({"timestamp": "2024-01-01 12:00:00+00", "metric": "test"})

    with pytest.raises(ValueError, match=r"Row must contain 'timestamp', 'metric', and 'value' fields"):
        writer.write({"metric": "test", "value": 1.0})

    writer.close()


# ============================================================================
# Metric Caching Tests
# ============================================================================


def test_metric_id_caching_behavior(test_writer: NormalizedNarrowSqlWriter, tsdb_engine: Engine):
    """
    Test metric ID caching reduces database lookups for repeated metric names.

    Validates that the in-memory cache correctly stores and retrieves metric IDs,
    minimizing database hits for repeated metric names while ensuring consistency
    across multiple writes.
    """
    # First write should populate cache
    first_data = {"timestamp": "2024-01-01 12:00:00+00", "metric": "accuracy", "value": 0.95}
    test_writer.write(first_data)

    # Verify metric is in cache
    assert "accuracy" in test_writer._metric_cache
    cached_id = test_writer._metric_cache["accuracy"]
    assert isinstance(cached_id, int)
    assert cached_id > 0

    # Second write with same metric should use cache
    second_data = {"timestamp": "2024-01-01 13:00:00+00", "metric": "accuracy", "value": 0.87}
    test_writer.write(second_data)

    # Cache should still contain the same ID
    assert test_writer._metric_cache["accuracy"] == cached_id

    # Verify both rows exist in the data table
    count = table_count(tsdb_engine, table_name="test_narrow_data", schema="public")
    assert count == 2, f"Expected 2 rows with accuracy metric, got {count}"


def test_cache_population_with_mixed_metrics(test_writer: NormalizedNarrowSqlWriter, sample_metric_data: list[dict]):
    """
    Test cache correctly handles multiple different metric names.

    Ensures the cache properly stores multiple metric IDs and that each
    metric name gets a unique ID that remains consistent across operations.
    """
    # Write batch with multiple metrics
    test_writer.write_many(sample_metric_data)

    # All metrics should be cached with unique IDs
    expected_metrics = {"accuracy", "loss", "f1_score"}
    assert_cache_contains_metrics(test_writer, expected_metrics)
    assert_unique_cached_ids(test_writer)

    # Verify all cached IDs are positive integers
    for metric_name in expected_metrics:
        assert_metric_cached(test_writer, metric_name)


def test_cache_persistence_across_writes(test_writer: NormalizedNarrowSqlWriter, tsdb_engine: Engine):
    """
    Test metric cache persists and remains accurate across multiple write operations.

    Validates that the cache doesn't get corrupted or cleared between operations
    and continues to provide consistent metric IDs for the same metric names.
    """
    # Write initial data to populate cache
    test_writer.write({"timestamp": "2024-01-01 12:00:00+00", "metric": "precision", "value": 0.92})
    initial_cache_size = len(test_writer._metric_cache)
    precision_id = test_writer._metric_cache["precision"]

    # Write new metric
    test_writer.write({"timestamp": "2024-01-01 13:00:00+00", "metric": "recall", "value": 0.89})

    # Cache should grow by one and preserve existing entry
    assert len(test_writer._metric_cache) == initial_cache_size + 1
    assert test_writer._metric_cache["precision"] == precision_id
    assert "recall" in test_writer._metric_cache

    # Reuse first metric
    test_writer.write({"timestamp": "2024-01-01 14:00:00+00", "metric": "precision", "value": 0.94})

    # Cache size should remain same, IDs should be consistent
    assert len(test_writer._metric_cache) == initial_cache_size + 1
    assert test_writer._metric_cache["precision"] == precision_id


# ============================================================================
# Error Handling & Edge Cases
# ============================================================================


def test_empty_batch_write(test_writer: NormalizedNarrowSqlWriter):
    """
    Test that empty batch writes are handled gracefully.

    Ensures the writer can handle empty sequences without errors,
    which is important for defensive programming in data processing pipelines.
    """
    # Empty list should not cause errors
    test_writer.write_many([])

    # Cache should remain empty
    assert len(test_writer._metric_cache) == 0


def test_special_characters_in_metric_names(
    test_writer: NormalizedNarrowSqlWriter,
    special_char_metrics: list[dict],
    tsdb_engine: Engine,
):
    """
    Test metric names with special characters are handled correctly.

    Validates that metric names containing spaces, punctuation, and Unicode
    characters are properly sanitized and stored in the database.
    """
    # Should handle special characters without errors
    test_writer.write_many(special_char_metrics)

    # Verify all metrics are stored correctly
    lookup_data = get_metric_lookup_data(tsdb_engine, "public", "test_narrow_metadata")
    stored_metrics = {row[1] for row in lookup_data}
    expected_metrics = {item["metric"] for item in special_char_metrics}

    assert expected_metrics.issubset(stored_metrics)

    # Verify data count
    expected_count = len(special_char_metrics)
    actual_count = table_count(tsdb_engine, table_name="test_narrow_data", schema="public")
    assert actual_count == expected_count, f"Expected {expected_count} rows, got {actual_count}"


def test_duplicate_metric_names_different_cases(
    test_writer: NormalizedNarrowSqlWriter,
    case_variant_metrics: list[dict],
):
    """
    Test that metric names are case-sensitive and treated as distinct.

    Ensures that 'Accuracy', 'accuracy', and 'ACCURACY' are treated as
    separate metrics with different IDs in the lookup table.
    """
    test_writer.write_many(case_variant_metrics)

    # All three should be cached as separate entries
    expected_metrics = {"accuracy", "Accuracy", "ACCURACY"}
    assert_cache_contains_metrics(test_writer, expected_metrics)
    assert len(test_writer._metric_cache) == 3

    # All should have different IDs
    assert_unique_cached_ids(test_writer)


def test_write_with_missing_optional_schema(tsdb_engine: Engine):
    """
    Test writer works correctly when no schema is specified.

    Validates that the writer can operate without an explicit schema,
    using the database's default schema for table creation and operations.
    """
    writer = NormalizedNarrowSqlWriter(
        engine=tsdb_engine,
        data_table_name="no_schema_data",
        lookup_table_name="no_schema_metrics",
        # schema intentionally omitted
    )

    test_data = {"timestamp": "2024-01-01 12:00:00+00", "metric": "no_schema_test", "value": 42.0}
    writer.write(test_data)

    # Verify data was written (should use public schema by default in PostgreSQL)
    count = table_count(tsdb_engine, table_name="no_schema_data", schema="public")
    assert count == 1, f"Expected 1 row in no_schema_data table, got {count}"

    writer.close()


# ============================================================================
# Table Creation & Schema Tests
# ============================================================================


def test_table_creation_on_first_write(writer_factory: Any, tsdb_engine: Engine):
    """
    Test lazy table creation occurs on first write operation.

    Validates that tables are created when needed and that the writer
    correctly detects when tables already exist to avoid recreation.
    """
    writer = writer_factory("lazy_create")

    # Tables should not exist initially
    assert not table_exists(tsdb_engine, table_name=writer.lookup_table_name, schema="public")
    assert not table_exists(tsdb_engine, table_name=writer.data_table_name, schema="public")

    # First write should trigger table creation
    test_data = {"timestamp": "2024-01-01 12:00:00+00", "metric": "lazy_test", "value": 1.0}
    writer.write(test_data)

    # Verify tables now exist in database
    assert table_exists(tsdb_engine, table_name=writer.lookup_table_name, schema="public")
    assert table_exists(tsdb_engine, table_name=writer.data_table_name, schema="public")


def test_table_schema_structure(writer_factory: Any, tsdb_engine: Engine):
    """
    Test that created tables have the correct schema structure.

    Validates column types, constraints, indexes, and foreign key relationships
    are properly established according to the writer's specifications.
    """
    writer = writer_factory("schema_test")

    # Trigger table creation
    writer.write({"timestamp": "2024-01-01 12:00:00+00", "metric": "schema_test", "value": 1.0})

    # Validate schema structure using assertion helpers
    assert_lookup_table_schema(tsdb_engine, "public", writer.lookup_table_name)
    assert_data_table_schema(tsdb_engine, "public", writer.data_table_name)


def test_existing_tables_detection(tsdb_engine: Engine):
    """
    Test writer correctly detects and uses existing tables.

    Ensures that if tables already exist (from a previous writer instance),
    the new writer instance correctly detects them and doesn't attempt recreation.
    """
    # First writer creates tables
    writer1 = NormalizedNarrowSqlWriter(
        engine=tsdb_engine,
        data_table_name="existing_test_data",
        lookup_table_name="existing_test_metrics",
        schema="public",
    )
    writer1.write({"timestamp": "2024-01-01 12:00:00+00", "metric": "existing_test", "value": 1.0})
    writer1.close()

    # Second writer should detect existing tables
    writer2 = NormalizedNarrowSqlWriter(
        engine=tsdb_engine,
        data_table_name="existing_test_data",
        lookup_table_name="existing_test_metrics",
        schema="public",
    )

    # This write should work without attempting table creation
    writer2.write({"timestamp": "2024-01-01 13:00:00+00", "metric": "existing_test2", "value": 2.0})

    # Verify both records exist
    count = table_count(tsdb_engine, table_name="existing_test_data", schema="public")
    assert count == 2, f"Expected 2 rows, got {count}"

    writer2.close()


# ============================================================================
# Resource Management Tests
# ============================================================================


def test_engine_disposal_on_close(tsdb_engine: Engine):
    """
    Test that engine is properly disposed when writer is closed.

    Validates proper resource cleanup to prevent connection leaks
    and ensure clean shutdown of database connections.
    """
    writer = NormalizedNarrowSqlWriter(
        engine=tsdb_engine,
        data_table_name="disposal_test_data",
        lookup_table_name="disposal_test_metrics",
        schema="public",
    )

    # Use the writer
    writer.write({"timestamp": "2024-01-01 12:00:00+00", "metric": "disposal_test", "value": 1.0})

    # Engine should be usable before close
    with tsdb_engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1

    # Close should dispose the engine
    writer.close()

    # After disposal, connections should fail
    try:
        with tsdb_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # If no exception, the engine wasn't disposed
        raise AssertionError("Engine was not disposed - connection still works")
    except Exception:
        # Expected: engine is disposed and connections fail
        pass


def test_flush_operation_is_noop(test_writer: NormalizedNarrowSqlWriter):
    """
    Test that flush operation is a no-op for this writer.

    The NormalizedNarrowSqlWriter writes immediately, so flush should
    not cause errors but also doesn't need to perform any operations.
    """
    # Flush on empty writer should work
    test_writer.flush()

    # Write some data
    test_writer.write({"timestamp": "2024-01-01 12:00:00+00", "metric": "flush_test", "value": 1.0})

    # Flush after write should also work
    test_writer.flush()
