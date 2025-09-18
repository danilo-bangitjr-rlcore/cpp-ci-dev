import random
import time
from collections.abc import Mapping
from types import MappingProxyType

import pytest
from corerl.sql_logging.sql_logging import SQLEngineConfig
from docker.models.containers import Container
from lib_sql.engine import get_sql_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy_utils.functions import drop_database

from test.infrastructure.utils.docker import init_docker_container


def wait_for_postgres_ready(container: Container, timeout: int = 60, check_interval: float = 1.0):
    """
    Wait for PostgreSQL container to be ready to accept connections.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Check if container is running
            container.reload()
            if container.status != 'running':
                time.sleep(check_interval)
                continue

            # Check PostgreSQL logs for ready signal
            logs = container.logs(tail=50).decode('utf-8')

            # Look for PostgreSQL ready indicators in logs
            ready_indicators = [
                'database system is ready to accept connections',
                'PostgreSQL init process complete',
                'listening on IPv4 address',
            ]

            if any(indicator in logs for indicator in ready_indicators):
                # Double-check with a simple connection test via container exec
                result = container.exec_run('pg_isready -h localhost -p 5432 -U postgres')
                if result.exit_code == 0:
                    return True

        except Exception:
            pass

        time.sleep(check_interval)

    container_id = container.id[:12] if container.id else "unknown"
    raise TimeoutError(f"PostgreSQL container {container_id} not ready after {timeout} seconds")


def init_postgres_container(
    name: str = "test_timescale",
    repository: str = "timescale/timescaledb",
    tag: str = "latest-pg17",
    port: int = 5433,
    restart: bool = True,
):
    """Initialize a PostgreSQL/TimescaleDB container with appropriate defaults."""

    env: Mapping[str, str] = MappingProxyType({"POSTGRES_PASSWORD": "password"})
    ports: Mapping[str, int] = MappingProxyType({"5432": port})

    container = init_docker_container(
        name=name,
        repository=repository,
        tag=tag,
        env=env,
        ports=ports,
        restart=restart,
    )

    # Wait for PostgreSQL to be ready before returning
    try:
        wait_for_postgres_ready(container, timeout=60)
    except TimeoutError as e:
        # Clean up container if it fails to start properly
        container.stop()
        container.remove()
        raise e

    return container


def wait_for_database_connection(engine: Engine, max_retries: int = 10, base_delay: float = 0.5):
    """
    Wait for database connection to be ready with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                # Simple test query to verify database is responsive
                conn.execute(text("SELECT 1"))
                return  # Success!
        except OperationalError as e:
            if attempt == max_retries - 1:
                raise e  # Last attempt failed, re-raise

            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
            time.sleep(delay)


@pytest.fixture(scope="module")
def tsdb_container(free_localhost_port: int):
    name = 'test_timescale' + str(random.randint(0, int(1e24)))
    container = init_postgres_container(name=name, port=free_localhost_port)
    yield container
    container.stop()
    container.remove()


@pytest.fixture(scope="function")
def tsdb_tmp_db_name(request: pytest.FixtureRequest) -> str:
    return request.node.nodeid + str(random.randint(0, int(1e24)))


@pytest.fixture(scope="function")
def tsdb_engine(tsdb_container: None, free_localhost_port: int, tsdb_tmp_db_name: str):
    cfg = SQLEngineConfig(
        drivername='postgresql+psycopg2',
        username='postgres',
        password='password',
        ip='localhost',
        port=free_localhost_port,
    )
    engine = get_sql_engine(cfg, tsdb_tmp_db_name)

    # Wait for database connection to be ready with retry logic
    wait_for_database_connection(engine, max_retries=15, base_delay=0.2)

    yield engine

    # Dispose of the engine to close all connections before dropping the database
    # This is required in SQLAlchemy 2.0 to prevent "database is being accessed by other users" errors
    engine.dispose()

    # Force disconnect all other sessions before dropping database
    # This handles cases where other engines might still be connected
    try:
        with get_sql_engine(cfg, "postgres").connect() as conn:
            # Terminate all active connections to the test database
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :db_name AND pid != pg_backend_pid()",
                ),
                {"db_name": tsdb_tmp_db_name},
            )
            conn.commit()
    except Exception:
        # If termination fails, proceed anyway - drop_database will handle the error
        pass

    drop_database(engine.url)
