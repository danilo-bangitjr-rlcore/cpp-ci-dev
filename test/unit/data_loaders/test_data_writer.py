from corerl.data_loaders.data_writer import DataWriter
from omegaconf import OmegaConf
from datetime import datetime, UTC
import subprocess
from pathlib import Path
import logging
import pytest
import time
import docker
from docker import DockerClient
from corerl.utils.docker import container_exists, stop_container

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

def start_timescale_container(client: DockerClient, name: str):
    
    create_timescale_container(client, name)
    container = client.containers.get(name)
    container.start()

@pytest.fixture(scope="module", autouse=True)
def timescale_docker():
    client = docker.from_env()
    container_name = "test_timescale"
    start_timescale_container(client, name=container_name)
    time.sleep(5) # give the container a few seconds to spin up
    yield
    
    # code after the yield is executed at cleanup time
    if INSPECT:
        return 
    
    stop_container(client, name=container_name)
    
    if PERSIST:
        return

    # prune container, deleting test data
    client.containers.prune()


def test_writing_datapt():
    # verify that the timescale container is running
    proc = subprocess.run(
        ["docker", "ps", "--filter", "name=assets-timescaledb-1", "--format", "{{.ID}}"], capture_output=True
    )
    out = proc.stdout
    assert out is not None

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

    data_writer = DataWriter(db_cfg=db_cfg, db_name="pytest")

    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)
