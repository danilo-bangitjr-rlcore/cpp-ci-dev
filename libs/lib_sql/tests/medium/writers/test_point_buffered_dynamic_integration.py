from __future__ import annotations

import time
from collections.abc import Callable
from typing import NamedTuple

import pytest
from sqlalchemy import Connection, Engine, text

from lib_sql.inspection import column_exists, table_exists
from lib_sql.writers.collectors.point_collecting_sql_writer import PointCollectingSqlWriter
from lib_sql.writers.core.dynamic_schema_sql_writer import DynamicSchemaSqlWriter
from lib_sql.writers.transforms.buffered_sql_writer import BufferedSqlWriter

pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.docker",
    "test.infrastructure.utils.tsdb",
]


class MetricRow(NamedTuple):
    """Test metrics row structure."""

    episode: int
    reward: float
    steps: int
    accuracy: float | None = None
    loss: float | None = None


def wait_for_database_state(
    engine: Engine,
    predicate: Callable[[Connection], bool],
    timeout: float = 2.0,
    interval: float = 0.05,
) -> bool:
    """Wait for database state to match predicate with proper connection management."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with engine.connect() as conn:
                if predicate(conn):
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


@pytest.fixture
def integration_table_name() -> str:
    """Unique table name for integration tests."""
    return "test_point_buffered_wide_integration"


@pytest.fixture
def dynamic_writer(tsdb_engine: Engine, integration_table_name: str) -> DynamicSchemaSqlWriter:
    """Create DynamicSchemaSqlWriter for integration testing."""
    return DynamicSchemaSqlWriter(
        engine=tsdb_engine,
        table_name=integration_table_name,
        table_creation_factory=lambda schema, table, columns: text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                {", ".join(f"{col.name} {col.type}" for col in columns)}
            );
        """),
        schema="public",
    )


@pytest.fixture
def buffered_writer(dynamic_writer: DynamicSchemaSqlWriter) -> BufferedSqlWriter:
    """Create BufferedSqlWriter wrapping DynamicSchemaSqlWriter."""
    return BufferedSqlWriter(
        inner=dynamic_writer,
        low_watermark=3,
        high_watermark=5,
        enabled=True,
    )


@pytest.fixture
def point_writer(buffered_writer: BufferedSqlWriter) -> PointCollectingSqlWriter:
    """Create PointCollectingSqlWriter wrapping BufferedSqlWriter wrapping DynamicSchemaSqlWriter."""
    return PointCollectingSqlWriter(
        inner=buffered_writer,
        row_factory=dict,
        enabled=True,
    )


@pytest.mark.timeout(10)
def test_basic_point_to_wide_integration(
    point_writer: PointCollectingSqlWriter,
    tsdb_engine: Engine,
    integration_table_name: str,
):
    """Test basic integration: collect points -> flush -> buffered -> dynamic schema storage."""
    # Initially no table should exist
    assert not table_exists(tsdb_engine, integration_table_name, schema="public")

    # Write individual metrics
    point_writer.write_point("episode", 100)
    point_writer.write_point("reward", 15.5)
    point_writer.write_point("steps", 250)

    # Flush to trigger buffered writer (but may not hit watermark yet)
    point_writer.flush()

    # Force flush by writing more data to hit high watermark
    for i in range(6):  # Should exceed high_watermark=5
        point_writer.write_point("episode", 100 + i)
        point_writer.write_point("reward", 15.5 + i)
        point_writer.write_point("steps", 250 + i)
        point_writer.flush()

    # Table should now exist
    assert wait_for_database_state(
        tsdb_engine,
        lambda conn: table_exists(tsdb_engine, integration_table_name, schema="public"),
        timeout=2.0,
    )

    # Verify columns were created
    assert column_exists(tsdb_engine, integration_table_name, "episode", schema="public")
    assert column_exists(tsdb_engine, integration_table_name, "reward", schema="public")
    assert column_exists(tsdb_engine, integration_table_name, "steps", schema="public")

    # Verify data was written
    def check_data_exists(conn: Connection) -> bool:
        result = conn.execute(text(f"SELECT COUNT(*) FROM public.{integration_table_name}"))
        count = result.scalar()
        return count is not None and count > 2

    assert wait_for_database_state(tsdb_engine, check_data_exists, timeout=2.0)

    point_writer.close()


@pytest.mark.timeout(10)
def test_watermark_behavior_through_point_collection(
    point_writer: PointCollectingSqlWriter,
    tsdb_engine: Engine,
    integration_table_name: str,
):
    """Test that buffered watermarks work correctly through point collection interface."""
    # Write data to reach low watermark (background flush)
    for i in range(3):  # low_watermark=3
        point_writer.write_point("episode", i)
        point_writer.write_point("reward", float(i))
        point_writer.flush()

    # Should trigger background flush, wait for table creation
    assert wait_for_database_state(
        tsdb_engine,
        lambda conn: table_exists(tsdb_engine, integration_table_name, schema="public"),
        timeout=2.0,
    )

    # Continue to high watermark (synchronous flush)
    for i in range(3, 6):  # Should hit high_watermark=5
        point_writer.write_point("episode", i)
        point_writer.write_point("reward", float(i))
        point_writer.flush()

    # Verify data was written by both background and synchronous flushes
    def check_multiple_rows(conn: Connection) -> bool:
        result = conn.execute(text(f"SELECT COUNT(*) FROM public.{integration_table_name}"))
        count = result.scalar()
        return count is not None and count >= 3

    assert wait_for_database_state(tsdb_engine, check_multiple_rows, timeout=2.0)

    point_writer.close()


@pytest.mark.timeout(10)
def test_dynamic_schema_evolution_through_composition(
    point_writer: PointCollectingSqlWriter,
    tsdb_engine: Engine,
    integration_table_name: str,
):
    """Test schema evolution when new metrics are introduced through the composition."""
    # Initial metrics
    point_writer.write_point("episode", 1)
    point_writer.write_point("reward", 10.0)
    point_writer.flush()

    # Add enough data to trigger flush and table creation
    for i in range(5):
        point_writer.write_point("episode", i + 2)
        point_writer.write_point("reward", 10.0 + i)
        point_writer.flush()

    # Wait for initial table
    assert wait_for_database_state(
        tsdb_engine,
        lambda conn: table_exists(tsdb_engine, integration_table_name, schema="public"),
        timeout=2.0,
    )

    # Verify initial columns
    assert column_exists(tsdb_engine, integration_table_name, "episode", schema="public")
    assert column_exists(tsdb_engine, integration_table_name, "reward", schema="public")
    assert not column_exists(tsdb_engine, integration_table_name, "accuracy", schema="public")

    # Add new metric types to trigger schema evolution
    point_writer.write_point("episode", 10)
    point_writer.write_point("reward", 20.0)
    point_writer.write_point("accuracy", 0.95)  # New metric
    point_writer.write_point("loss", 0.05)  # Another new metric
    point_writer.flush()

    # Force additional flushes to ensure schema evolution
    for i in range(4):
        point_writer.write_point("accuracy", 0.9 + i * 0.01)
        point_writer.write_point("loss", 0.05 + i * 0.01)
        point_writer.flush()

    # Verify new columns were added
    assert wait_for_database_state(
        tsdb_engine,
        lambda conn: column_exists(tsdb_engine, integration_table_name, "accuracy", schema="public"),
        timeout=2.0,
    )

    assert column_exists(tsdb_engine, integration_table_name, "loss", schema="public")

    point_writer.close()


@pytest.mark.timeout(10)
def test_disabled_writer_composition(
    tsdb_engine: Engine,
    integration_table_name: str,
    dynamic_writer: DynamicSchemaSqlWriter,
):
    """Test that disabled point collector properly disables the entire composition."""
    # Create composition with disabled point collector
    buffered = BufferedSqlWriter(dynamic_writer, low_watermark=2, high_watermark=4)
    disabled_point_writer = PointCollectingSqlWriter(
        inner=buffered,
        row_factory=dict,
        enabled=False,
    )

    # Write data that would normally trigger flushes
    for i in range(10):
        disabled_point_writer.write_point("episode", i)
        disabled_point_writer.write_point("reward", float(i))
        disabled_point_writer.flush()

    # Wait a moment to ensure no background operations
    time.sleep(0.2)

    # Table should not exist since writer is disabled
    assert not table_exists(tsdb_engine, integration_table_name, schema="public")

    disabled_point_writer.close()


@pytest.mark.timeout(10)
def test_close_behavior_and_resource_cleanup(
    point_writer: PointCollectingSqlWriter,
    tsdb_engine: Engine,
    integration_table_name: str,
):
    """Test that close() properly flushes data and cleans up resources."""
    # Add some data without reaching watermarks
    point_writer.write_point("episode", 42)
    point_writer.write_point("reward", 100.0)
    point_writer.write_point("steps", 500)

    # Close should flush pending data through the entire composition
    point_writer.close()

    # Verify data was flushed and written despite not hitting watermarks
    assert wait_for_database_state(
        tsdb_engine,
        lambda conn: table_exists(tsdb_engine, integration_table_name, schema="public"),
        timeout=2.0,
    )

    def check_data_written(conn: Connection) -> bool:
        result = conn.execute(text(f"SELECT COUNT(*) FROM public.{integration_table_name}"))
        count = result.scalar()
        return count is not None and count > 0

    assert wait_for_database_state(tsdb_engine, check_data_written, timeout=2.0)


@pytest.mark.timeout(10)
def test_empty_flushes_and_partial_data(
    point_writer: PointCollectingSqlWriter,
    tsdb_engine: Engine,
    integration_table_name: str,
):
    """Test behavior with empty flushes and partial metric sets."""
    # Empty flush should be no-op
    point_writer.flush()
    time.sleep(0.1)
    assert not table_exists(tsdb_engine, integration_table_name, schema="public")

    # Partial data with missing metrics
    point_writer.write_point("episode", 1)
    # Note: only episode, no reward or steps
    point_writer.flush()

    # Add more data to trigger actual flush
    for i in range(5):
        point_writer.write_point("episode", i + 2)
        if i % 2 == 0:  # Only add reward sometimes
            point_writer.write_point("reward", float(i + 2))
        point_writer.flush()

    # Should still work with partial data
    assert wait_for_database_state(
        tsdb_engine,
        lambda conn: table_exists(tsdb_engine, integration_table_name, schema="public"),
        timeout=2.0,
    )

    assert column_exists(tsdb_engine, integration_table_name, "episode", schema="public")

    point_writer.close()
