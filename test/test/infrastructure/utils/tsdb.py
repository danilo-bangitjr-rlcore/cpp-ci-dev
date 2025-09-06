import random
import time

import pytest
from corerl.sql_logging.sql_logging import SQLEngineConfig
from lib_sql.engine import get_sql_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy_utils.functions import drop_database

from test.infrastructure.utils.docker import init_docker_container


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
    container = init_docker_container(name=name, ports={"5432": free_localhost_port})
    yield container
    container.stop()
    container.remove()


@pytest.fixture(scope="function")
def tsdb_tmp_db_name(request: pytest.FixtureRequest) -> str:
    return request.node.nodeid


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
    drop_database(engine.url)
