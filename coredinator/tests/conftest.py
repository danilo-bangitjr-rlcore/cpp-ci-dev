import os
import shutil
import socket
import stat
import subprocess
from pathlib import Path

import psutil
import pytest

from coredinator.utils.process import terminate_process_tree
from tests.utils.factories import create_dummy_config
from tests.utils.service_fixtures import CoredinatorService, wait_for_service_healthy


@pytest.fixture()
def free_localhost_port():
    """Function-scoped fixture to get a free port for each test."""
    # Binding to port 0 will ask the OS to give us an arbitrary free port
    # since we've just bound that free port, it is by definition no longer free,
    # so we set that port as reuseable to allow another socket to bind to it
    # then we immediately close the socket and release our connection.
    sock = socket.socket()
    sock.bind(("localhost", 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    port: int = sock.getsockname()[1]
    sock.close()

    return port


@pytest.fixture()
def long_running_agent_env(monkeypatch: pytest.MonkeyPatch):
    """Fixture to configure fake agents to run in long-running mode."""
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Fixture to create a temporary configuration file."""
    cfg = tmp_path / "example_config.yaml"
    create_dummy_config(cfg)
    return cfg


@pytest.fixture()
def dist_with_fake_executable(tmp_path: Path) -> Path:
    """Fixture to create a temporary directory with fake executables."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    src = Path(__file__).resolve().parent / "fixtures" / "fake_agent.py"
    dst_coreio = dist_dir / "coreio-1.0.0"
    dst_corerl = dist_dir / "corerl-1.0.0"
    for dst in [dst_coreio, dst_corerl]:
        shutil.copy(src, dst)
        # Make executable
        mode = dst.stat().st_mode
        dst.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dist_dir


@pytest.fixture()
def coredinator_service(dist_with_fake_executable: Path, free_localhost_port: int, monkeypatch: pytest.MonkeyPatch):
    """Fixture to start coredinator service as subprocess for e2e testing.

    Returns information about the running service including process ID for advanced control.
    """
    # Ensure fake agents stay alive when started by coredinator
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    port = free_localhost_port
    base_url = f"http://localhost:{port}"

    # Start coredinator using the documented approach from README with custom port
    cmd = [
        "uv",
        "run",
        "python",
        "coredinator/app.py",
        "--base-path",
        str(dist_with_fake_executable),
        "--port",
        str(port),
    ]

    # Set environment for subprocess
    env = dict(os.environ, FAKE_AGENT_BEHAVIOR="long")
    cwd = Path(__file__).parent.parent  # Run from coredinator package root

    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for service to start (try healthcheck)
    wait_for_service_healthy(base_url, process=process)

    service_info = CoredinatorService(
        base_url=base_url,
        process_id=process.pid,
        command=cmd,
        env=env,
        cwd=cwd,
    )

    yield service_info

    # Attempt clean shutdown - handle case where process was already terminated
    try:
        proc = psutil.Process(process.pid)
        terminate_process_tree(proc, timeout=5.0)
    except psutil.NoSuchProcess:
        # Process was already terminated (e.g., by test code)
        pass
