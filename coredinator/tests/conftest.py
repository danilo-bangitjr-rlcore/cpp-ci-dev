import os
import shutil
import socket
import stat
import subprocess
from pathlib import Path

import pytest
from lib_process.process import Process
from lib_process.process_list import find_processes_by_name_patterns
from PyInstaller.__main__ import run as pyinstaller_run

from tests.utils.factories import create_dummy_config
from tests.utils.service_fixtures import CoredinatorService, wait_for_service_healthy
from tests.utils.timeout_multiplier import apply_timeout_multiplier, get_timeout_multiplier

FAKE_AGENT_BASENAMES: tuple[str, str] = ("coreio-1.0.0", "corerl-1.0.0")


def _platform_executable_name(base_name: str) -> str:
    return f"{base_name}.exe" if os.name == "nt" else base_name


@pytest.fixture(scope="session")
def fake_agent_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the fake agent as a standalone executable using PyInstaller once per test session."""

    build_root = tmp_path_factory.mktemp("fake-agent-build")
    dist_dir = build_root / "dist"
    work_dir = build_root / "build"
    spec_dir = build_root / "spec"
    src = Path(__file__).resolve().parent / "fixtures" / "fake_agent.py"

    pyinstaller_args = [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "fake_agent",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
    ]
    if os.name == "nt":
        pyinstaller_args.append("--noconsole")
    pyinstaller_args.append(str(src))

    pyinstaller_run(pyinstaller_args)

    executable_name = _platform_executable_name("fake_agent")
    executable_path = dist_dir / executable_name
    if not executable_path.exists():
        raise FileNotFoundError(f"PyInstaller did not create expected executable at {executable_path}")

    return executable_path


@pytest.fixture()
def free_localhost_port():
    """Function-scoped fixture to get a free port for each test."""
    # Binding to port 0 will ask the OS to give us an arbitrary free port
    # since we've just bound that free port, it is by definition no longer free,
    # so we set that port as reusable to allow another socket to bind to it
    # then we immediately close the socket and release our connection.
    sock = socket.socket()
    sock.bind(("localhost", 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    port: int = sock.getsockname()[1]
    sock.close()

    return port


@pytest.fixture()
def platform_timeout_multiplier():
    return get_timeout_multiplier()


@pytest.fixture()
def adjusted_timeout():
    return apply_timeout_multiplier


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
def dist_with_fake_executable(tmp_path: Path, fake_agent_binary: Path) -> Path:
    """Fixture to create a temporary directory with fake executables."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    for base_name in FAKE_AGENT_BASENAMES:
        dst = dist_dir / _platform_executable_name(base_name)
        shutil.copy(fake_agent_binary, dst)
        if os.name != "nt":
            # Ensure executable bit is set when running on POSIX systems
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
    log_file = dist_with_fake_executable.parent / "coredinator.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

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

    cmd.extend(["--log-file", str(log_file)])

    # Set environment for subprocess
    env = dict(os.environ, FAKE_AGENT_BEHAVIOR="long")
    cwd = Path(__file__).parent.parent  # Run from coredinator package root

    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for service to start (try healthcheck)
    wait_for_service_healthy(base_url, process=process, log_file=log_file)

    service_info = CoredinatorService(
        base_url=base_url,
        process_id=process.pid,
        command=cmd,
        env=env,
        cwd=cwd,
        log_file=log_file,
    )

    yield service_info

    proc = Process.from_pid(process.pid)
    proc.terminate_tree(timeout=5.0)


@pytest.fixture(scope="session", autouse=True)
def cleanup_lingering_processes():
    """Session-scoped fixture to clean up any lingering coreio/corerl processes at the end of tests."""

    yield

    processes = find_processes_by_name_patterns(["coreio", "corerl", "fake_agent"])
    for proc in processes:
        proc.terminate_tree(timeout=apply_timeout_multiplier(2.0))
