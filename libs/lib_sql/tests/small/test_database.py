from pathlib import Path

import pytest
from sqlalchemy import URL
from sqlalchemy_utils import database_exists

from lib_sql.database import maybe_create_database, maybe_drop_database


@pytest.fixture
def temp_db_url(tmp_path: Path):
    """Fixture for a temporary SQLite database URL."""
    db_path = tmp_path / "temp.db"
    url = URL.create(drivername="sqlite", database=str(db_path))
    yield url
    # Cleanup: drop if exists
    if database_exists(url):
        maybe_drop_database(url)


@pytest.fixture
def existing_db_url(temp_db_url: URL) -> URL:
    """Fixture for an existing SQLite database URL."""
    maybe_create_database(temp_db_url)
    return temp_db_url


def test_maybe_drop_database_nonexistent(temp_db_url: URL):
    """
    Test dropping a non-existent database does nothing.
    """
    assert not database_exists(temp_db_url)
    maybe_drop_database(temp_db_url)  # Should not raise
    assert not database_exists(temp_db_url)


def test_maybe_drop_database_existing(existing_db_url: URL):
    """
    Test dropping an existing database.
    """
    assert database_exists(existing_db_url)
    maybe_drop_database(existing_db_url)
    assert not database_exists(existing_db_url)


def test_maybe_create_database_nonexistent(temp_db_url: URL):
    """
    Test creating a non-existent database.
    """
    assert not database_exists(temp_db_url)
    maybe_create_database(temp_db_url, backoff_seconds=0, max_tries=1)
    assert database_exists(temp_db_url)


def test_maybe_create_database_existing(existing_db_url: URL):
    """
    Test creating an existing database does nothing.
    """
    assert database_exists(existing_db_url)
    maybe_create_database(existing_db_url, backoff_seconds=0, max_tries=1)
    assert database_exists(existing_db_url)


# For failure case, using invalid URL
@pytest.fixture
def invalid_db_url() -> URL:
    """
    Fixture for an invalid database URL.
    """
    return URL.create(drivername="sqlite", database="/invalid/path/that/cannot/be/created")


def test_maybe_create_database_failure(invalid_db_url: URL):
    """
    Test database creation failure after retries.
    """
    with pytest.raises(Exception, match="database creation failed"):
        maybe_create_database(invalid_db_url, backoff_seconds=0, max_tries=1)
