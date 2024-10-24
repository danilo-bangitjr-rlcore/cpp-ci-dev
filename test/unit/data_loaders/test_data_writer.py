from sqlalchemy import text, Engine
from corerl.data_loaders.data_writer import DataWriter
from omegaconf import OmegaConf
from datetime import datetime, UTC
import pytest
import time
import docker
from docker import DockerClient
from corerl.utils.docker import container_exists, stop_container
from corerl.sql_logging.sql_logging import table_exists

PERSIST = False  # if true, stop container but don't remove it (data will persist)
INSPECT = False  # if true, leave container running after tests conclude


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


# generate a create table statement to reflect an existing table with
# pg_dump -h your_host -U your_user -p your_port your_database -t your_table --schema-only
# example:
# pg_dump -h localhost -U postgres -p 5432 postgres -t mock_system --schema-only

def start_timescale_container(client: DockerClient, name: str):
    create_timescale_container(client, name)
    container = client.containers.get(name)
    container.start()


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
    with engine.connect() as connection:
        connection.execute(text(create_table_stmt))
        connection.commit()

    # TODO: execute statements to make this a hypertable


def maybe_create_sensor_table(engine: Engine, sensor_table_name: str):
    if table_exists(engine, table_name=sensor_table_name):
        print("table exists")
        return

    create_sensor_table(engine, sensor_table_name)


@pytest.fixture(scope="module", autouse=True)
def timescale_docker():
    client = docker.from_env()
    container_name = "test_timescale"
    start_timescale_container(client, name=container_name)
    # TODO: poll for connection instead of hard coded wait
    time.sleep(3)  # give the container a few seconds to spin up
    yield

    # code after the yield is executed at cleanup time
    if INSPECT:
        return

    stop_container(client, name=container_name)

    if PERSIST:
        return

    # prune container, deleting test data
    client.containers.prune()


@pytest.mark.skip(reason="github actions do not yet support docker")
def test_writing_datapt():
    # connect to the db
    db_cfg = OmegaConf.create(
        {
            "drivername": "postgresql+psycopg2",
            "username": "postgres",
            "password": "password",
            "ip": "localhost",
            "port": 5433,  # default is 5432, but we want to use different port for test db
        }
    )

    db_name = "pytest"
    sensor_table_name = "sensors"
    data_writer = DataWriter(db_cfg=db_cfg, db_name=db_name, sensor_table_name=sensor_table_name)
    maybe_create_sensor_table(engine=data_writer.engine, sensor_table_name=sensor_table_name)

    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)
