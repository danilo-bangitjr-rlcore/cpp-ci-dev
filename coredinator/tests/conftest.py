import os
import shutil
import socket
import stat
import subprocess
import time
from pathlib import Path

import psutil
import pytest
import requests


@pytest.fixture()
def free_localhost_port():
    """Function-scoped fixture to get a free port for each test."""
    # Binding to port 0 will ask the OS to give us an arbitrary free port
    # since we've just bound that free port, it is by definition no longer free,
    # so we set that port as reuseable to allow another socket to bind to it
    # then we immediately close the socket and release our connection.
    sock = socket.socket()
    sock.bind(('localhost', 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    port: int = sock.getsockname()[1]
    sock.close()

    return port


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Fixture to create a temporary configuration file."""
    cfg = tmp_path / "example_config.yaml"
    cfg.write_text("dummy: true\n")
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

    Returns the base URL where the service is running.
    """
    # Ensure fake agents stay alive when started by coredinator
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    port = free_localhost_port
    base_url = f"http://localhost:{port}"

    # Start coredinator using the documented approach from README with custom port
    cmd = [
        "uv", "run", "python", "coredinator/app.py",
        "--base-path", str(dist_with_fake_executable),
        "--port", str(port),
    ]

    # Set environment for subprocess
    env = dict(os.environ, FAKE_AGENT_BEHAVIOR="long")

    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=Path(__file__).parent.parent,  # Run from coredinator package root
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for service to start (try healthcheck)
    max_attempts = 30
    for _ in range(max_attempts):
        time.sleep(0.1)
        try:
            response = requests.get(f"{base_url}/api/healthcheck", timeout=1)
            if response.status_code == 200:
                break
        except requests.RequestException:
            ...

    else:
        # Service didn't start - kill process and fail
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        error_msg = (
            f"Coredinator service failed to start on {base_url}\n" +
            f"stdout: {stdout.decode()}\n" +
            f"stderr: {stderr.decode()}"
        )
        raise RuntimeError(error_msg)

    try:
        yield base_url
    finally:
        # Attempt clean shutdown
        proc = psutil.Process(process.pid)
        for child in proc.children(recursive=True):
            child.terminate()

        proc.terminate()

        # Shutdown forcefully if needed
        _, alive = psutil.wait_procs([proc, *proc.children(recursive=True)], timeout=5)
        for p in alive:
            p.kill()
