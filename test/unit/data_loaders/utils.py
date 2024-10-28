from corerl.sql_logging.sql_logging import table_exists
from sqlalchemy import text, Engine
import pytest
from corerl.utils.docker import container_exists, stop_container
from docker import DockerClient
import time
import docker

PERSIST = False  # if true, stop container but don't remove it (data will persist)
INSPECT = False  # if true, leave container running after tests conclude

def maybe_create_sensor_table(engine: Engine, sensor_table_name: str):
    if table_exists(engine, table_name=sensor_table_name):
        print("table exists")
        return

    create_sensor_table(engine, sensor_table_name)

# generate a create table statement to reflect an existing table with
# pg_dump -h your_host -U your_user -p your_port your_database -t your_table --schema-only
# example:
# pg_dump -h localhost -U postgres -p 5432 postgres -t mock_system --schema-only
def create_sensor_table(engine: Engine, sensor_table_name: str):
    create_table_stmt = f"""
        CREATE TABLE public.{sensor_table_name} (
            "time" timestamp with time zone NOT NULL,
            host text,
            id text,
            name text,
            "Quality" text,
            fields jsonb
        );
    """
    create_hypertable_stmt = f"""
        SELECT create_hypertable('{sensor_table_name}', 'time', chunk_time_interval => INTERVAL '1h');
    """
    print(create_table_stmt)
    print(create_hypertable_stmt)
    with engine.connect() as connection:
        connection.execute(text(create_table_stmt))
        connection.execute(text(create_hypertable_stmt))
        connection.commit()

def create_timescale_container(client: DockerClient, name: str) -> None:
    if container_exists(client, name):
        return

    env = {"POSTGRES_PASSWORD": "password"}
    client.containers.create(
        image="timescale/timescaledb-ha:pg16",
        detach=True,
        ports={"5432": 5433},
        environment=env,
        name=name,
    )

def start_timescale_container(client: DockerClient, name: str):
    create_timescale_container(client, name)
    container = client.containers.get(name)
    container.start()


@pytest.fixture(scope="session", autouse=True)
def timescale_docker():
    client = docker.from_env()
    container_name = "test_timescale"
    start_timescale_container(client, name=container_name)
    # TODO: poll for connection instead of hard coded wait
    yield

    # code after the yield is executed at cleanup time
    if INSPECT:
        return

    stop_container(client, name=container_name)

    if PERSIST:
        return

    # prune container, deleting test data
    client.containers.prune()
