"""Unit tests for SqlReader with mocked database."""

from unittest.mock import Mock, patch

import pytest
from coretelemetry.utils.sql import DBConfig, SqlReader
from pytest import MonkeyPatch
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError


@pytest.fixture
def mock_engine() -> Mock:
    """Mock SQLAlchemy engine."""
    return Mock(spec=Engine)


@pytest.fixture
def mock_db_config() -> DBConfig:
    """Mock DBConfig with test values."""
    return DBConfig(
        username="test_user",
        password="test_pass",
        ip="localhost",
        port=5432,
        db_name="test_db",
        schema="public",
    )


@pytest.fixture
def sql_reader(mock_db_config: DBConfig, mock_engine: Mock, monkeypatch: MonkeyPatch) -> SqlReader:
    """SqlReader with mocked engine."""
    monkeypatch.setattr("coretelemetry.utils.sql.get_sql_engine", lambda **kwargs: mock_engine)
    return SqlReader(mock_db_config)

class TestBuildQuery:
    """Tests for build_query method."""

    def test_build_query_no_time_params(self, sql_reader: SqlReader) -> None:
        """Test build_query with no time parameters adds LIMIT 1."""
        query, params = sql_reader.build_query(
            table_name="metrics",
            column_name="temperature",
            start_time=None,
            end_time=None,
            time_col=True,
            not_null=False,
        )

        assert "SELECT time, temperature FROM metrics" in query
        assert "LIMIT 1" in query
        assert params == {}

    def test_build_query_start_time_only(self, sql_reader: SqlReader) -> None:
        """Test build_query with only start_time."""
        query, params = sql_reader.build_query(
            table_name="metrics",
            column_name="temperature",
            start_time="2024-01-01",
            end_time=None,
            time_col=True,
            not_null=False,
        )

        assert "WHERE time >= :start_time" in query
        assert "LIMIT 1" not in query
        assert params == {"start_time": "2024-01-01"}

    def test_build_query_end_time_only(self, sql_reader: SqlReader) -> None:
        """Test build_query with only end_time."""
        query, params = sql_reader.build_query(
            table_name="metrics",
            column_name="temperature",
            start_time=None,
            end_time="2024-01-02",
            time_col=True,
            not_null=False,
        )

        assert "WHERE time <= :end_time" in query
        assert "LIMIT 1" not in query
        assert params == {"end_time": "2024-01-02"}

    def test_build_query_both_time_params(self, sql_reader: SqlReader) -> None:
        """Test build_query with both start_time and end_time."""
        query, params = sql_reader.build_query(
            table_name="metrics",
            column_name="temperature",
            start_time="2024-01-01",
            end_time="2024-01-02",
            time_col=True,
            not_null=False,
        )

        assert "WHERE time >= :start_time AND time <= :end_time" in query
        assert "LIMIT 1" not in query
        assert params == {"start_time": "2024-01-01", "end_time": "2024-01-02"}

    def test_build_query_with_not_null(self, sql_reader: SqlReader) -> None:
        """Test build_query with not_null flag."""
        query, _params = sql_reader.build_query(
            table_name="metrics",
            column_name="temperature",
            start_time=None,
            end_time=None,
            time_col=True,
            not_null=True,
        )

        assert "WHERE temperature IS NOT NULL" in query
        assert "LIMIT 1" in query

    def test_build_query_without_time_column(self, sql_reader: SqlReader) -> None:
        """Test build_query without time column."""
        query, _params = sql_reader.build_query(
            table_name="metrics",
            column_name="temperature",
            start_time=None,
            end_time=None,
            time_col=False,
            not_null=False,
        )

        assert "SELECT temperature FROM metrics" in query
        assert "time" not in query.split("FROM")[0]


class TestTestConnection:
    """Tests for test_connection method."""

    def test_test_connection_success(self, sql_reader: SqlReader, monkeypatch: MonkeyPatch) -> None:
        """Test test_connection returns True on successful connection."""
        mock_connection = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_connection)
        mock_context_manager.__exit__ = Mock(return_value=False)

        with patch("coretelemetry.utils.sql.TryConnectContextManager", return_value=mock_context_manager):
            result = sql_reader.test_connection()

        assert result is True
        mock_connection.execute.assert_called_once()

    def test_test_connection_failure(self, sql_reader: SqlReader, monkeypatch: MonkeyPatch) -> None:
        """Test test_connection returns False on connection failure."""
        mock_context_manager = Mock()
        error = OperationalError("Connection failed", params=None, orig=Exception("Connection failed"))
        mock_context_manager.__enter__ = Mock(side_effect=error)

        with patch("coretelemetry.utils.sql.TryConnectContextManager", return_value=mock_context_manager):
            result = sql_reader.test_connection()

        assert result is False
