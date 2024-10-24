from corerl.data_loaders.data_writer import DataWriter
from omegaconf import OmegaConf
from datetime import datetime, UTC
import subprocess
from pathlib import Path
import logging
import pytest
import time

PERSIST = False  # if true, stop container but don't remove it (data will persist)
INSPECT = False  # if true, leave container running after tests conclude

@pytest.fixture(scope="module", autouse=True)
def docker_compose_up():
    compose_path = Path.cwd() / Path("test/integration/assets/compose.yaml")
    logging.info(compose_path)

    proc = subprocess.run(["docker", "compose", "-f", f"{compose_path}", "up", "timescaledb", "-d"])
    time.sleep(5)  # give docker a few seconds to spin up db

    yield
    
    # code after the yield is executed at cleanup time

    if not PERSIST:
        # here we stop and remove the container used in the test
        subprocess.run(["docker", "compose", "-f", f"{compose_path}", "down", "timescaledb"])
        return_code = proc.returncode
        print(f"docker compose down return code: {return_code}")
        proc.check_returncode()
    elif not INSPECT:
        # here we just stop the container, but don't remove it
        # the data will persist if the container is restarted
        subprocess.run(["docker", "container", "stop", "assets-timescaledb-1"])
        return_code = proc.returncode
        print(f"docker container stop return code: {return_code}")
        proc.check_returncode()




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
