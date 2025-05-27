import random

import pytest
from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine
from sqlalchemy_utils.functions import drop_database

from test.infrastructure.utils.docker import init_docker_container


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
    yield engine

    drop_database(engine.url)
