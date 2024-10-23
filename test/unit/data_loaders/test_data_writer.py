from corerl.data_loaders.data_writer import DataWriter
from datetime import datetime, UTC
import subprocess
from pathlib import Path
import logging
import pytest


@pytest.fixture(scope="module", autouse=True)
def docker_compose_up():
    compose_path = Path.cwd() / Path("test/integration/assets/compose.yaml")
    logging.info(compose_path)

    proc = subprocess.run(['docker', 'compose', '-f', f'{compose_path}', 'up', 'timescaledb', '-d'])
    yield

    subprocess.run(['docker', 'compose', '-f', f'{compose_path}', 'down', 'timescaledb'])
    return_code = proc.returncode
    logging.info(return_code)
    proc.check_returncode()


def test_writing_datapt():
    proc = subprocess.run(["docker", "ps", "--filter", "name=assets-timescaledb-1", "--format", "{{.ID}}"], capture_output=True)

    out = proc.stdout
    print(out) # for debugging if test fails
    assert out is not None
    
    data_writer = DataWriter(username="postgres", password="password", db_name="sensors")

    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)
