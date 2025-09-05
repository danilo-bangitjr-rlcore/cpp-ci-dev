from typing import NamedTuple

import pandas as pd
import pytest
from lib_utils.sql_logging.connect_engine import TryConnectContextManager
from sqlalchemy import Engine, text

from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig, sanitize_keys


class BasicDummyPoint(NamedTuple):
    id: int
    name: str
    value: float
    extra_field: str = "default"

class ExtendedDummyPoint(NamedTuple):
    id: int
    name: str
    value: float
    extra_field: str
    bonus_column: float

DummyPoint = BasicDummyPoint | ExtendedDummyPoint

@pytest.fixture()
def dummy_writer_config(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    port = tsdb_engine.url.port
    assert port is not None

    return BufferedWriterConfig(
        enabled=True,
        drivername='postgresql+psycopg2',
        username='postgres',
        password='password',
        ip='localhost',
        port=port,
        db_name=tsdb_tmp_db_name,
        table_schema='public',
        static_columns=False,
        table_name='test_buffered_table',
    )

class DummyBufferedWriter(BufferedWriter[DummyPoint]):
    def __init__(self, cfg: BufferedWriterConfig):
        super().__init__(cfg)

    def _create_table_sql(self):
        return text(f"""
            CREATE TABLE {self.cfg.table_schema}.{self.cfg.table_name} (
                id INTEGER,
                name TEXT,
                value FLOAT,
                extra_field TEXT
            )
        """)

    def write(self, point: DummyPoint):
        self._write(point)

    def read(self):
        assert self.engine is not None
        with TryConnectContextManager(self.engine) as conn:
            return pd.read_sql_table('test_buffered_table', con=conn)

@pytest.fixture()
def dummy_writer(dummy_writer_config: BufferedWriterConfig):
    writer = DummyBufferedWriter(dummy_writer_config)
    yield writer
    writer.close()


def test_add_column(
    tsdb_engine: Engine,
    dummy_writer: DummyBufferedWriter,
):
    """Test that new columns are dynamically added to the table."""
    # Write initial data with base columns
    dummy_writer.ensure_table_exists()
    dummy_writer.blocking_sync()

    # Verify initial table structure
    dummy_writer.ensure_known_columns_initialized()
    expected_initial_columns = {'id', 'name', 'value', 'extra_field'}
    assert dummy_writer._known_columns == expected_initial_columns

    dummy_writer.add_columns(['new_col'])

    # Verify new columns were added
    with tsdb_engine.connect() as conn:
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'test_buffered_table' AND table_schema = 'public'
            ORDER BY column_name
        """))
        final_columns = {row[0] for row in result}

    expected_final_columns = {'id', 'name', 'value', 'extra_field', 'new_col'}
    assert final_columns == expected_final_columns


def test_column_preservation_on_addition(
    dummy_writer: DummyBufferedWriter,
):
    """Test that existing data is preserved when new columns are added."""
    # Write multiple rows with existing columns
    initial_data = [
        BasicDummyPoint(id=1, name="first", value=10.0, extra_field="a"),
        BasicDummyPoint(id=2, name="second", value=20.0, extra_field="b"),
        BasicDummyPoint(id=3, name="third", value=30.0, extra_field="c"),
    ]

    for point in initial_data:
        dummy_writer.write(point)
    dummy_writer.blocking_sync()

    # Add a new point with an extra column
    extended_point = ExtendedDummyPoint(id=4, name="fourth", value=40.0, extra_field="d", bonus_column=99.9)
    dummy_writer.write(extended_point)
    dummy_writer.blocking_sync()

    # Verify original data is preserved
    final_df = dummy_writer.read()

    assert len(final_df) == 4

    # Check against expected DataFrame structure
    expected_df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['first', 'second', 'third', 'fourth'],
        'value': [10.0, 20.0, 30.0, 40.0],
        'extra_field': ['a', 'b', 'c', 'd'],
        'bonus_column': [None, None, None, 99.9],
    })

    pd.testing.assert_frame_equal(final_df, expected_df)


def test_key_sanitization(
):
    """Test that column names are properly sanitized."""
    # Create data with problematic column names
    problematic_data = [
        {'id': 1, 'my-column': 10.0, 'another.column': 20.0, 'column with spaces': 30.0},
        {'id': 2, 'my-column': 15.0, 'weird@#$%column': 25.0, 'column___multiple___underscores': 35.0},
    ]

    # Manually trigger the sanitization process
    sanitized_data = sanitize_keys(problematic_data)

    # Verify sanitization
    assert 'my_column' in sanitized_data[0]
    assert 'another_column' in sanitized_data[0]
    assert 'column_with_spaces' in sanitized_data[0]
    assert 'weird_column' in sanitized_data[1]
    assert 'column_multiple_underscores' in sanitized_data[1]

    # Verify no problematic characters remain
    for data_dict in sanitized_data:
        for key in data_dict.keys():
            assert '-' not in key
            assert '.' not in key
            assert ' ' not in key
            assert '@' not in key
            assert '#' not in key
            assert '$' not in key
            assert '%' not in key
            # Check no multiple consecutive underscores
            assert '___' not in key


def test_static_columns_mode(
    dummy_writer_config: BufferedWriterConfig,
    caplog: pytest.LogCaptureFixture,
):
    # Enable static columns mode
    dummy_writer_config.static_columns = True
    writer = DummyBufferedWriter(dummy_writer_config)

    # Write initial data with base columns
    initial_point = BasicDummyPoint(id=1, name="test", value=10.0, extra_field="initial")
    writer.write(initial_point)
    writer.blocking_sync()

    # Attempt to write data with new columns - should be filtered out
    extended_point = ExtendedDummyPoint(id=2, name="extended", value=20.0, extra_field="second", bonus_column=99.9)
    writer.write(extended_point)
    writer.blocking_sync()

    # Verify data was written but without the new column
    final_df = writer.read()

    # Verify only 2 rows and expected structure
    assert len(final_df) == 2

    # Create expected DataFrame for comparison. Note no bonus column
    expected_df = pd.DataFrame({
        'id': [1, 2],
        'name': ['test', 'extended'],
        'value': [10.0, 20.0],
        'extra_field': ['initial', 'second'],
    })

    pd.testing.assert_frame_equal(final_df, expected_df)
    writer.close()
