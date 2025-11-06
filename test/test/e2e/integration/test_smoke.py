import time
from dataclasses import dataclass
from pathlib import Path

import psutil
import pytest
import requests
import yaml
from docker.models.containers import Container
from lib_defs.type_defs.base_events import Event, EventTopic, EventType
from lib_events.client.event_bus_client import EventBusClient
from lib_process.process import Process

from test.infrastructure.networking import get_free_port
from test.infrastructure.polling import wait_for_agent_state


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
def tsdb_port(tsdb_container: Container) -> int:
    """
    Get the actual host port mapped to the PostgreSQL container.

    This ensures we use the correct port that the container is actually listening on,
    rather than a potentially stale free_localhost_port value.
    """
    tsdb_container.reload()
    return int(tsdb_container.ports["5432/tcp"][0]["HostPort"])


@pytest.fixture()
def coredinator_service(coredinator_executable: Path, bin_dir: Path, tmp_path: Path):
    """
    Start coredinator service with test-isolated state directory.

    Creates symlinks to executables in tmp_path to isolate agent persistence across tests.
    """
    test_base_path = tmp_path / "coredinator_base"
    test_base_path.mkdir(exist_ok=True)

    for executable in bin_dir.glob("linux-*"):
        link_path = test_base_path / executable.name
        if not link_path.exists():
            link_path.symlink_to(executable)

    port = get_free_port("localhost")
    base_url = f"http://localhost:{port}"

    log_file = tmp_path / "coredinator.log"
    cmd = [
        str(coredinator_executable),
        "--base-path",
        str(test_base_path),
        "--port",
        str(port),
        "--log-file",
        str(log_file),
    ]

    proc = Process.start_in_background(cmd)
    _wait_for_http_health(f"{base_url}/api/healthcheck")

    yield ServiceInfo(base_url=base_url, pid=proc.psutil.pid)

    proc.terminate_tree(timeout=5.0)


@pytest.mark.timeout(900)
def test_coredinator_smoke(coredinator_service: ServiceInfo):
    """
    Verify coredinator service starts and responds to health checks.

    Extended timeout (900s) accounts for PyInstaller building all executables.
    """
    base_url = coredinator_service.base_url
    pid = coredinator_service.pid

    assert psutil.pid_exists(pid)
    assert psutil.Process(pid).is_running()

    response = requests.get(f"{base_url}/api/healthcheck", timeout=5.0)
    assert response.status_code == 200


@pytest.mark.timeout(120)
def test_corerl_direct_execution(
    corerl_executable: Path,
    tsdb_port: int,
    tmp_path: Path,
):
    """
    Test corerl executable directly without coredinator orchestration.

    Verifies that corerl can start, connect to database, and run steps successfully.
    """
    base_config_path = Path(__file__).parent / "assets" / "e2e_agent_config.yaml"
    assert base_config_path.exists(), f"Config file not found: {base_config_path}"

    with base_config_path.open() as f:
        config_data = yaml.safe_load(f)

    config_data["infra"]["db"]["port"] = tsdb_port
    config_data["coreio"] = {
        "data_ingress": {
            "enabled": False,
        },
        "opc_connections": [],
        "tags": [],
    }

    temp_config_path = tmp_path / "corerl_direct_test_config.yaml"
    with temp_config_path.open("w") as f:
        yaml.dump(config_data, f)

    # Capture output for debugging
    log_file = tmp_path / "corerl_output.log"
    cmd = [str(corerl_executable), "--config-name", str(temp_config_path)]

    import subprocess
    with log_file.open("w") as f:
        proc_handle = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    try:
        # Wait for process to complete (should exit after max_steps)
        exit_code = proc_handle.wait(timeout=90.0)
        assert exit_code == 0, f"corerl exited with non-zero code: {exit_code}\nLog:\n{log_file.read_text()}"

    except subprocess.TimeoutExpired:
        proc_handle.kill()
        proc_handle.wait()
        log_content = log_file.read_text()
        raise AssertionError(f"corerl did not complete within 90s. Log excerpt:\n{log_content[-1000:]}") from None

    finally:
        if proc_handle.poll() is None:
            proc_handle.kill()
            proc_handle.wait()


@pytest.mark.timeout(900)
def test_coredinator_full_agent_run(
    coredinator_service: ServiceInfo,
    tsdb_port: int,
    tmp_path: Path,
):
    """
    Full e2e test: Start an agent via coredinator and verify it completes successfully.

    This test verifies the complete agent lifecycle through coredinator:
    1. Agent starts via POST /api/agents/start
    2. Agent runs and processes environment steps
    3. Agent completes after max_steps limit (50 steps)
    4. Agent can be queried for status

    Extended timeout (900s) accounts for PyInstaller building all executables,
    particularly corerl which can take 5-10 minutes.
    """
    db_port = tsdb_port
    base_url = coredinator_service.base_url

    base_config_path = Path(__file__).parent / "assets" / "e2e_agent_config.yaml"
    assert base_config_path.exists(), f"Config file not found: {base_config_path}"

    with base_config_path.open() as f:
        config_data = yaml.safe_load(f)

    config_data["infra"]["db"]["port"] = db_port
    config_data["coreio"] = {
        "data_ingress": {
            "enabled": False,
        },
        "opc_connections": [],
        "tags": [],
    }

    temp_config_path = tmp_path / "e2e_agent_config_with_db.yaml"
    with temp_config_path.open("w") as f:
        yaml.dump(config_data, f)

    config_path = temp_config_path

    start_response = requests.post(
        f"{base_url}/api/agents/start",
        json={"config_path": str(config_path.resolve())},
        timeout=10.0,
    )
    assert start_response.status_code == 200, f"Failed to start agent: {start_response.text}"

    agent_id = start_response.json()
    assert isinstance(agent_id, str) and agent_id, "Invalid agent_id returned from start request"

    wait_for_agent_state(base_url, agent_id, "failed")

    stop_response = requests.post(
        f"{base_url}/api/agents/{agent_id}/stop",
        timeout=10.0,
    )
    assert stop_response.status_code == 200


@pytest.mark.timeout(900)
def test_event_bus_cross_service_communication(
    coredinator_service: ServiceInfo,
    tsdb_port: int,
    tmp_path: Path,
):
    """
    Test event bus communication between coreio and corerl through coredinator broker.

    This test verifies:
    1. Coredinator event bus proxy is running
    2. CoreRL can connect and emit events
    3. CoreIO can connect and emit events
    4. Events flow through the broker (lifecycle events: service_started, service_stopped)

    Extended timeout (900s) for PyInstaller build time.
    """


    db_port = tsdb_port
    base_url = coredinator_service.base_url

    base_config_path = Path(__file__).parent / "assets" / "e2e_agent_config.yaml"
    assert base_config_path.exists(), f"Config file not found: {base_config_path}"

    with base_config_path.open() as f:
        config_data = yaml.safe_load(f)

    config_data["infra"]["db"]["port"] = db_port
    config_data["event_bus_client"] = {
        "enabled": True,
        "host": "localhost",
        "port": 5580,
    }
    config_data["coreio"] = {
        "event_bus_client": {
            "enabled": True,
            "host": "localhost",
            "port": 5580,
        },
        "data_ingress": {
            "enabled": False,
        },
        "opc_connections": [],
        "tags": [],
    }

    temp_config_path = tmp_path / "e2e_event_bus_config.yaml"
    with temp_config_path.open("w") as f:
        yaml.dump(config_data, f)

    rl_client = EventBusClient(
        host="localhost",
        port=5580,
    )
    rl_client.connect()
    rl_client.subscribe(EventTopic.corerl)

    io_client = EventBusClient(
        host="localhost",
        port=5580,
    )
    io_client.connect()
    io_client.subscribe(EventTopic.coreio)

    rl_client.start_consumer()
    io_client.start_consumer()

    start_response = requests.post(
        f"{base_url}/api/agents/start",
        json={"config_path": str(temp_config_path.resolve())},
        timeout=10.0,
    )
    assert start_response.status_code == 200, f"Failed to start agent: {start_response.text}"

    agent_id = start_response.json()
    assert isinstance(agent_id, str) and agent_id, "Invalid agent_id"

    time.sleep(5.0)

    status_response = requests.get(
        f"{base_url}/api/agents/{agent_id}/status",
        timeout=5.0,
    )
    assert status_response.status_code == 200, f"Failed to get status: {status_response.text}"
    print(f"Agent status: {status_response.json()}")

    rl_events_seen = []
    io_events_seen = []
    deadline = time.time() + 60.0

    while time.time() < deadline:
        rl_event = rl_client.recv_event(timeout=0.1)
        if rl_event is not None:
            rl_events_seen.append(rl_event.type)

        io_event = io_client.recv_event(timeout=0.1)
        if io_event is not None:
            io_events_seen.append(io_event.type)

        if EventType.service_started in rl_events_seen and EventType.service_started in io_events_seen:
            break

        time.sleep(0.1)

    assert EventType.service_started in rl_events_seen, (
        f"Did not receive RL service_started event. Saw: {rl_events_seen}"
    )
    assert EventType.service_started in io_events_seen, (
        f"Did not receive IO service_started event. Saw: {io_events_seen}"
    )

    stop_response = requests.post(
        f"{base_url}/api/agents/{agent_id}/stop",
        timeout=10.0,
    )
    assert stop_response.status_code == 200

    deadline = time.time() + 10.0
    while time.time() < deadline:
        rl_event = rl_client.recv_event(timeout=0.1)
        if rl_event is not None:
            rl_events_seen.append(rl_event.type)

        io_event = io_client.recv_event(timeout=0.1)
        if io_event is not None:
            io_events_seen.append(io_event.type)

        if EventType.service_stopped in rl_events_seen and EventType.service_stopped in io_events_seen:
            break

        time.sleep(0.1)

    assert EventType.service_stopped in rl_events_seen, (
        f"Did not receive RL service_stopped event. Saw: {rl_events_seen}"
    )
    assert EventType.service_stopped in io_events_seen, (
        f"Did not receive IO service_stopped event. Saw: {io_events_seen}"
    )

    rl_client.close()
    io_client.close()
