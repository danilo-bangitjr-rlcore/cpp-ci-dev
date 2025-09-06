from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from lib_sql.connection import TryConnectContextManager, try_connect


@pytest.fixture
def temp_db_url(tmp_path: Path) -> URL:
    """Fixture for a temporary SQLite database URL."""
    db_path = tmp_path / "test.db"
    return URL.create(drivername="sqlite", database=str(db_path))


@pytest.fixture
def working_engine(temp_db_url: URL) -> Engine:
    """Fixture for a working SQLite engine."""
    return create_engine(temp_db_url)


@pytest.fixture
def failing_engine() -> Engine:
    """Fixture for an engine that will fail to connect."""
    invalid_url = URL.create(drivername="sqlite", database="/invalid/read-only/path/test.db")
    return create_engine(invalid_url)


class TestTryConnect:
    """Test the try_connect function core behavior."""

    def test_successful_connection(self, working_engine: Engine):
        """Test successful connection on first attempt."""
        conn = try_connect(working_engine, backoff_seconds=0, max_tries=1)
        assert conn is not None
        assert not conn.closed
        conn.close()

    def test_connection_failure_exceeds_max_tries(self, failing_engine: Engine):
        """Test connection failure when max tries is exceeded."""
        with pytest.raises(Exception, match="sql engine connection failed"):
            try_connect(failing_engine, backoff_seconds=1, max_tries=2)

    @pytest.mark.parametrize("exception_type", [
        sqlite3.OperationalError("Database locked"),
        ConnectionError("Connection refused"),
        SQLAlchemyError("General SQL error"),
    ])
    def test_retry_on_various_exceptions(self, exception_type: Exception):
        """Test that various exception types trigger retries."""
        mock_engine = Mock(spec=Engine)
        mock_engine.connect.side_effect = [exception_type, Mock()]

        conn = try_connect(mock_engine, backoff_seconds=0, max_tries=2)
        assert conn is not None
        assert mock_engine.connect.call_count == 2


class TestTryConnectContextManager:
    """Test the TryConnectContextManager essential functionality."""

    def test_successful_context_manager_flow(self, working_engine: Engine):
        """Test successful enter/exit flow with connection cleanup."""
        with TryConnectContextManager(working_engine, backoff_seconds=0, max_tries=1) as conn:
            assert conn is not None
            assert not conn.closed
        assert conn.closed

    def test_context_manager_connection_failure(self, failing_engine: Engine):
        """Test context manager when connection fails."""
        with pytest.raises(Exception, match="sql engine connection failed"):
            with TryConnectContextManager(failing_engine, backoff_seconds=0, max_tries=1):
                pass

    def test_context_manager_exception_handling(self, working_engine: Engine):
        """Test context manager cleanup when exception occurs within block."""
        connection_ref = None
        with pytest.raises(ValueError, match="test exception"):
            with TryConnectContextManager(working_engine, backoff_seconds=0, max_tries=1) as conn:
                connection_ref = conn
                assert not conn.closed
                raise ValueError("test exception")

        assert connection_ref is not None
        assert connection_ref.closed

    def test_context_manager_parameter_storage(self, working_engine: Engine):
        """Test context manager stores and uses initialization parameters."""
        cm = TryConnectContextManager(working_engine, backoff_seconds=2, max_tries=5)
        assert cm.engine == working_engine
        assert cm.backoff_seconds == 2
        assert cm.max_tries == 5
        assert cm.conn is None
