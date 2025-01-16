import subprocess
from typing import Any, Generator

import pytest
from docker.models.containers import Container

from test.infrastructure.utils.docker import init_docker_container


@pytest.fixture(scope="module")
def init_data_reader_tsdb_container():
    container = init_docker_container(ports={"5432": 5435})
    yield container
    container.stop()
    container.remove()

@pytest.mark.timeout(240)
def test_deployment_manager(init_data_reader_tsdb_container: Generator[Container, Any, None]):
    """
    Should be able to execute the deployment script with the saturation experiment
    without error. If an error code is returned (i.e. the process crashes),
    then test fails.

    This test does no checking of result validity.
    """

    proc = subprocess.run([
        'uv', 'run',
        'python', 'deployment_manager.py',
        '--base', 'test/large/smoke/assets',
        '--config-name', 'deployment_manager',
        'deployment.python_executable="uv run python"'
    ])
    proc.check_returncode()
