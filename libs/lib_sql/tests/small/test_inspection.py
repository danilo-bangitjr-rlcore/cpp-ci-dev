from pathlib import Path

import pytest
from sqlalchemy import URL, Column, DateTime, Integer, MetaData, String, Table, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from lib_sql.inspection import (
    column_exists,
    get_all_columns,
    get_column_type,
    table_exists,
)


@pytest.fixture
def temp_db_url(tmp_path: Path):
    db_path = tmp_path / "test_rl_inspection.db"
    return URL.create(drivername="sqlite", database=str(db_path))


@pytest.fixture
def populated_engine(temp_db_url: URL):
    engine = create_engine(temp_db_url)
    metadata = MetaData()

    Table(
        "metrics",
        metadata,
        Column("time", DateTime, nullable=False),
        Column("agent_step", Integer, nullable=False),
        Column("metric", String(100), nullable=False),
        Column("value", String(50)),
    )

    Table(
        "metrics_wide",
        metadata,
        Column("time", DateTime, nullable=False),
        Column("agent_step", Integer, nullable=False),
        Column("reward", String(20)),
        Column("loss", String(20)),
        Column("episode_length", Integer),
    )

    Table(
        "exp-data",
        metadata,
        Column("exp-id", Integer),
        Column("agent_name", String(50)),
        Column("env.name", String(100)),
    )

    metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def empty_engine(temp_db_url: URL):
    engine = create_engine(temp_db_url)
    yield engine
    engine.dispose()


class TestTableExistsIntegration:
    """Integration tests for table_exists() using real SQLite RL metrics database."""

    @pytest.mark.timeout(5)
    @pytest.mark.parametrize("table_name,expected", [
        ("metrics", True),
        ("metrics_wide", True),
        ("exp-data", True),
        ("nonexistent", False),
        ("Metrics", False),  # Case sensitive
    ])
    def test_table_exists_real_database(self, populated_engine: Engine, table_name: str, expected: bool):
        """
        Test table_exists with real RL metrics database tables.
        """
        result = table_exists(populated_engine, table_name)
        assert result == expected

    @pytest.mark.timeout(5)
    def test_table_exists_empty_database(self, empty_engine: Engine):
        """
        Test table_exists with empty database.
        """
        assert table_exists(empty_engine, "any_table") is False

    @pytest.mark.timeout(5)
    def test_table_exists_schema_parameter(self, populated_engine: Engine):
        """
        Test table_exists with schema parameter (SQLite ignores schemas).
        """
        assert table_exists(populated_engine, "metrics", schema=None) is True


class TestColumnExistsIntegration:
    """Integration tests for column_exists() using real SQLite RL metrics database."""

    @pytest.mark.parametrize("table_name,column_name,expected", [
        ("metrics", "time", True),
        ("metrics", "agent_step", True),
        ("metrics", "metric", True),
        ("metrics", "value", True),
        ("metrics", "nonexistent", False),
        ("metrics_wide", "reward", True),
        ("metrics_wide", "loss", True),
        ("exp-data", "exp-id", True),
        ("exp-data", "agent_name", True),
        ("exp-data", "env.name", True),
        ("nonexistent_table", "any_column", False),
    ])
    def test_column_exists_real_database(
        self, populated_engine: Engine, table_name: str, column_name: str, expected: bool,
    ):
        """
        Test column_exists with real RL metrics database tables and columns.
        """
        result = column_exists(populated_engine, table_name, column_name)
        assert result == expected

    @pytest.mark.timeout(5)
    def test_column_exists_sqlalchemy_error_handling(self, empty_engine: Engine):
        """
        Test column_exists gracefully handles SQLAlchemy errors.
        """
        result = column_exists(empty_engine, "nonexistent_table", "any_column")
        assert result is False


class TestGetColumnTypeIntegration:
    """Integration tests for get_column_type() using real SQLite RL metrics database."""

    def test_get_column_type_various_types(self, populated_engine: Engine):
        """
        Test get_column_type returns correct SQLAlchemy types.
        """
        time_type = get_column_type(populated_engine, "metrics", "time")
        agent_step_type = get_column_type(populated_engine, "metrics", "agent_step")
        metric_type = get_column_type(populated_engine, "metrics", "metric")
        value_type = get_column_type(populated_engine, "metrics", "value")

        assert time_type is not None
        assert agent_step_type is not None
        assert metric_type is not None
        assert value_type is not None

    def test_get_column_type_special_characters(self, populated_engine: Engine):
        """
        Test get_column_type with special character column names.
        """
        exp_id_type = get_column_type(populated_engine, "exp-data", "exp-id")
        env_name_type = get_column_type(populated_engine, "exp-data", "env.name")

        assert exp_id_type is not None
        assert env_name_type is not None

    def test_get_column_type_nonexistent_column(self, populated_engine: Engine):
        """
        Test get_column_type raises assertion for nonexistent column.
        """
        with pytest.raises(AssertionError, match="SQL Error, column not found"):
            get_column_type(populated_engine, "metrics", "nonexistent_column")


class TestGetAllColumnsIntegration:
    """Integration tests for get_all_columns() using real SQLite RL metrics database."""

    def test_get_all_columns_metrics_table(self, populated_engine: Engine):
        """
        Test get_all_columns returns complete column metadata for metrics table.
        """
        columns = get_all_columns(populated_engine, "metrics")

        assert len(columns) == 4
        column_names = [col["name"] for col in columns]
        assert "time" in column_names
        assert "agent_step" in column_names
        assert "metric" in column_names
        assert "value" in column_names

        for column in columns:
            assert "name" in column
            assert "type" in column
            assert "nullable" in column

    def test_get_all_columns_special_table(self, populated_engine: Engine):
        """
        Test get_all_columns with table and columns having special characters.
        """
        columns = get_all_columns(populated_engine, "exp-data")

        assert len(columns) == 3
        column_names = [col["name"] for col in columns]
        assert "exp-id" in column_names
        assert "agent_name" in column_names
        assert "env.name" in column_names

    def test_get_all_columns_empty_table_list(self, empty_engine: Engine):
        """
        Test get_all_columns with nonexistent table raises exception.
        """
        with pytest.raises(SQLAlchemyError):
            get_all_columns(empty_engine, "nonexistent_table")


class TestInspectionEdgeCases:
    """Test edge cases and error conditions for inspection functions."""

    def test_engine_disposal_handling(self, temp_db_url: URL):
        """
        Test inspection functions work with properly disposed engines.
        """
        engine = create_engine(temp_db_url)
        metadata = MetaData()
        Table("test", metadata, Column("id", Integer))
        metadata.create_all(engine)

        assert table_exists(engine, "test") is True
        assert column_exists(engine, "test", "id") is True

        engine.dispose()

        new_engine = create_engine(temp_db_url)
        assert table_exists(new_engine, "test") is True
        assert column_exists(new_engine, "test", "id") is True
        new_engine.dispose()

    def test_schema_parameter_consistency(self, populated_engine: Engine):
        """
        Test all functions handle schema parameter consistently.
        """
        assert table_exists(populated_engine, "metrics", schema=None) is True
        assert column_exists(populated_engine, "metrics", "time", schema=None) is True
        assert get_column_type(populated_engine, "metrics", "time", schema=None) is not None
        assert len(get_all_columns(populated_engine, "metrics", schema=None)) > 0
