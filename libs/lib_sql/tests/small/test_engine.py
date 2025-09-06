from pathlib import Path

import pytest
from sqlalchemy import URL, text
from sqlalchemy.engine import Engine

from lib_sql.engine import (
    get_sql_engine,
    try_create_engine,
)


class RealSQLEngineConfig:
    def __init__(
        self,
        drivername: str = "sqlite",
        username: str = "",
        password: str = "",
        ip: str = "",
        port: int = 0,
    ):
        self.drivername = drivername
        self.username = username
        self.password = password
        self.ip = ip
        self.port = port


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_engine.db")


@pytest.fixture
def sqlite_url(temp_db_path: str) -> URL:
    return URL.create(drivername="sqlite", database=temp_db_path)


@pytest.fixture
def invalid_url() -> URL:
    return URL.create(drivername="nonexistent_driver", database="test.db")


@pytest.fixture
def sqlite_config(temp_db_path: str) -> RealSQLEngineConfig:
    return RealSQLEngineConfig(drivername="sqlite")


class TestTryCreateEngine:
    @pytest.mark.timeout(5)
    def test_successful_engine_creation(self, sqlite_url: URL):
        """
        Test successful engine creation.
        """
        engine = try_create_engine(sqlite_url, backoff_seconds=0, max_tries=1)
        assert engine is not None
        assert isinstance(engine, Engine)
        assert engine.url.drivername == "sqlite"

    @pytest.mark.timeout(5)
    def test_engine_creation_failure_exceeds_max_tries(self, invalid_url: URL):
        """
        Test engine creation failure when max tries is exceeded.
        """
        with pytest.raises(Exception, match="sql engine creation failed"):
            try_create_engine(invalid_url, backoff_seconds=0, max_tries=2)


class TestGetSqlEngine:
    @pytest.mark.timeout(10)
    def test_get_sql_engine_success_flow(self, sqlite_config: RealSQLEngineConfig, temp_db_path: str):
        """
        Test successful engine creation with database operations.
        """
        engine = get_sql_engine(sqlite_config, temp_db_path, force_drop=False)
        assert engine is not None
        assert isinstance(engine, Engine)
        assert engine.url.database == temp_db_path

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @pytest.mark.timeout(10)
    def test_get_sql_engine_with_force_drop(self, tmp_path: Path):
        """
        Test engine creation with force_drop enabled.
        """
        db_path = str(tmp_path / "drop_test.db")
        config = RealSQLEngineConfig(drivername="sqlite")

        engine1 = get_sql_engine(config, db_path, force_drop=False)
        with engine1.connect() as conn:
            conn.execute(text("CREATE TABLE test_table (id INTEGER)"))
            conn.commit()
        engine1.dispose()

        engine2 = get_sql_engine(config, db_path, force_drop=True)
        with engine2.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = result.fetchall()
            assert len([t for t in tables if not t[0].startswith("sqlite_")]) == 0
        engine2.dispose()

    @pytest.mark.timeout(5)
    def test_config_to_url_construction(self):
        """
        Test URL construction from config protocol.
        """
        config = RealSQLEngineConfig(
            drivername="postgresql",
            username="testuser",
            password="testpass",
            ip="127.0.0.1",
            port=5432,
        )

        from lib_sql.engine import sqlalchemy

        url_object = sqlalchemy.URL.create(
            drivername=config.drivername,
            username=config.username,
            password=config.password,
            host=config.ip,
            port=config.port,
            database="test_db",
        )

        assert str(url_object).startswith("postgresql://testuser:")
        assert "127.0.0.1:5432" in str(url_object)
        assert str(url_object).endswith("test_db")

    @pytest.mark.timeout(5)
    def test_engine_failure_propagation(self):
        """
        Test that engine creation failures are propagated.
        """
        from lib_sql.engine import try_create_engine

        url_object = URL.create(
            drivername="nonexistent_driver",
            username="",
            password="",
            host="",
            port=0,
            database="test_db",
        )

        with pytest.raises(Exception, match="sql engine creation failed"):
            try_create_engine(url_object, backoff_seconds=0, max_tries=1)
