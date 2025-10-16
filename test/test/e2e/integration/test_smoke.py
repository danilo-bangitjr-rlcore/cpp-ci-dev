import time
from dataclasses import dataclass
from pathlib import Path

import psutil
import pytest
import requests
from lib_process.process import Process

from test.infrastructure.networking import get_free_port


@dataclass
class ServiceInfo:
    base_url: str
    pid: int


def _wait_for_http_health(url: str, timeout: float = 10.0) -> None:
    """
    Poll HTTP endpoint until it returns 200 or timeout expires.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Endpoint {url} did not become healthy within {timeout}s")


@pytest.fixture()
def bin_dir(
    coredinator_executable: Path,
    coreio_executable: Path,
    corerl_executable: Path,
) -> Path:
    """
    Ensure all executables are built and return the bin directory.

    This fixture depends on all three executable fixtures to trigger their builds.
    """
    return coredinator_executable.parent


@pytest.fixture()
def coredinator_service(coredinator_executable: Path, bin_dir: Path, tmp_path: Path):
    """
    Start coredinator service with executables in bin directory.

    Coredinator will use the bin directory as base_path to find coreio/corerl executables.
    """
    port = get_free_port("localhost")
    base_url = f"http://localhost:{port}"

    cmd = [
        str(coredinator_executable),
        "--base-path",
        str(bin_dir),
        "--port",
        str(port),
    ]

    proc = Process.start_in_background(cmd)
    _wait_for_http_health(f"{base_url}/api/healthcheck")

    yield ServiceInfo(base_url=base_url, pid=proc.psutil.pid)

    proc.terminate_tree(timeout=5.0)


@pytest.mark.timeout(300)
def test_coredinator_smoke(coredinator_service: ServiceInfo):
    """
    Verify coredinator service starts and responds to health checks.
    """
    base_url = coredinator_service.base_url
    pid = coredinator_service.pid

    assert psutil.pid_exists(pid)
    assert psutil.Process(pid).is_running()

    response = requests.get(f"{base_url}/api/healthcheck", timeout=5.0)
    assert response.status_code == 200
