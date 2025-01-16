import subprocess
from typing import Any, Generator

import pytest
from docker.models.containers import Container

from test.infrastructure.utils.docker import init_docker_container


@pytest.fixture(scope="module")
def init_data_reader_tsdb_container():
    container = init_docker_container(ports={"5432": 5436})
    yield container
    container.stop()
    container.remove()


@pytest.mark.parametrize('config', [
    'pendulum',
    'saturation',
    'mountain_car_continuous',
])
@pytest.mark.timeout(120)
def test_main_configs(init_data_reader_tsdb_container: Generator[Container, Any, None], config: str):
    """
    Should be able to execute the main script for several configs
    without error. If an error code is returned (i.e. the process crashes),
    then test fails.

    This test does no checking of result validity.
    """

    proc = subprocess.run([
        'uv', 'run', 'python', 'main.py',
        '--config-name', f'{config}', 'experiment.max_steps=25',
        'metrics.port=5436',
    ])
    proc.check_returncode()
