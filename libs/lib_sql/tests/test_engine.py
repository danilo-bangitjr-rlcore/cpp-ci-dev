from pathlib import Path

import pytest
from sqlalchemy import URL

from lib_sql.database import maybe_create_database
from lib_sql.engine import get_sql_engine, try_create_engine


class TestConfig:
    """Test config for SQLite."""
    drivername = "sqlite"
    username = ""
    password = ""
    ip = ""
    port = 0


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Fixture for a temporary database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def valid_url(temp_db_path: str) -> URL:
    """Fixture for a valid SQLite URL."""
    return URL.create(drivername="sqlite", database=temp_db_path)


def test_try_create_engine_success(valid_url: URL):
    """
    Test successful engine creation.
    """
    engine = try_create_engine(valid_url, backoff_seconds=0, max_tries=1)
    assert engine is not None


def test_get_sql_engine_with_force_drop(temp_db_path: str):
    """
    Test get_sql_engine with force_drop.
    """
    config = TestConfig()
    # First create
    maybe_create_database(URL.create(drivername="sqlite", database=temp_db_path))
    engine = get_sql_engine(config, temp_db_path, force_drop=True)
    assert engine is not None
