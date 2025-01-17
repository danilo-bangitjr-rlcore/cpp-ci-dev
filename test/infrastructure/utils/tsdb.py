import random

import pytest
from sqlalchemy import Engine, text

from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine
from test.infrastructure.utils.docker import init_docker_container


@pytest.fixture(scope="module")
def tsdb_container(free_localhost_port: int):
    container = init_docker_container(ports={"5432": free_localhost_port})
    yield container
    container.stop()
    container.remove()


@pytest.fixture(scope="function")
def tsdb_test_db_name():
    return ''.join(random.choices('abcdefghijk', k=8))


@pytest.fixture(scope="function")
def tsdb_engine(tsdb_container: None, free_localhost_port: int, tsdb_test_db_name: str):
    cfg = SQLEngineConfig(
        port=free_localhost_port,
    )
    engine = get_sql_engine(cfg, tsdb_test_db_name)
    return engine


@pytest.fixture(scope="function")
def tmp_table_name(tsdb_engine: Engine):
    table_name = ''.join(random.choices('abcdefghijk', k=8))
    yield table_name

    with tsdb_engine.connect() as conn:
        conn.execute(text(f'DROP TABLE {table_name};'))
