"""Large integration tests for coredinator service.

These tests start the coredinator service as a subprocess and make real HTTP
requests to test end-to-end workflows.
"""

import sqlite3
import subprocess
from pathlib import Path

import psutil
import pytest
import requests

from coredinator.test_utils import CoredinatorService, wait_for_service_healthy
from coredinator.utils.process import terminate_process_tree, wait_for_termination
from coredinator.utils.test_polling import wait_for_event


def get_microservice_pids(base_path: Path, agent_id: str):
    """Fetch corerl and coreio process IDs for an agent from the persistence database."""
    db_path = base_path / "agent_state.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT corerl_process_id, coreio_process_id FROM agent_states WHERE agent_id = ?",
            (agent_id,),
        )
        row = cursor.fetchone()
        if row:
            corerl_pid, coreio_pid = row
            return {"corerl": corerl_pid, "coreio": coreio_pid}
        return {"corerl": None, "coreio": None}


@pytest.mark.timeout(15)
def test_agent_status_unknown_returns_stopped(coredinator_service: CoredinatorService):
    """Test that querying status for unknown agent returns 'stopped' state."""
    base_url = coredinator_service.base_url
    unknown_agent_id = "nonexistent-agent"

    # Query status for an agent that doesn't exist
    response = requests.get(f"{base_url}/api/agents/{unknown_agent_id}/status")

    # Should return 200 with stopped state
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == unknown_agent_id
    assert data["state"] == "stopped"
    assert data["config_path"] is None


@pytest.mark.timeout(30)
def test_microservice_failure_recovery(
    coredinator_service: CoredinatorService,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """Simulate microservice failure and verify coredinator recovers agent state."""
    base_url = coredinator_service.base_url
    agent_id = config_file.stem

    # Start the agent
    response = requests.post(f"{base_url}/api/agents/start", json={"config_path": str(config_file)})
    assert response.status_code == 200, f"Failed to start agent: {response.text}"

    # Wait for the microservice PIDs to be recorded in the database
    def _coreio_pid_available():
        pids = get_microservice_pids(dist_with_fake_executable, agent_id)
        return pids["coreio"] is not None

    assert wait_for_event(_coreio_pid_available, interval=0.1, timeout=2.0), "coreio PID should be present in DB"

    pids = get_microservice_pids(dist_with_fake_executable, agent_id)
    coreio_pid = pids["coreio"]

    # Simulate microservice failure
    proc = psutil.Process(coreio_pid)
    proc.terminate()

    assert wait_for_termination(proc, timeout=5.0, poll_interval=0.1), "Process did not terminate in time"

    # Query agent status after microservice failure
    response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "failed"


@pytest.mark.timeout(30)
def test_coredinator_failure_recovery(coredinator_service: CoredinatorService, config_file: Path):
    """Simulate coredinator failure and verify agent state is restored after restart."""
    # Start the agent
    agent_id = config_file.stem
    response = requests.post(f"{coredinator_service.base_url}/api/agents/start", json={"config_path": str(config_file)})
    assert response.status_code == 200, f"Failed to start agent: {response.text}"

    # Wait for agent to be running before simulating coredinator failure
    def _agent_running():
        response = requests.get(f"{coredinator_service.base_url}/api/agents/{agent_id}/status")
        return response.status_code == 200 and response.json().get("state") == "running"

    assert wait_for_event(_agent_running, interval=0.1, timeout=2.0), \
        "Agent should be running before coredinator failure test"

    # Kill coredinator process to simulate failure
    proc_obj = psutil.Process(coredinator_service.process_id)
    assert terminate_process_tree(proc_obj, timeout=5.0), "Coredinator did not terminate in time"

    # Restart coredinator service using the same configuration
    proc2 = subprocess.Popen(
        coredinator_service.command,
        env=coredinator_service.env,
        cwd=coredinator_service.cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for restarted service to start
    wait_for_service_healthy(coredinator_service.base_url, process=proc2)

    # Query agent status after restart
    response = requests.get(f"{coredinator_service.base_url}/api/agents/{agent_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == agent_id
    assert data["state"] in ("running", "stopped")

    # Cleanup the restarted process
    try:
        proc_obj = psutil.Process(proc2.pid)
        terminate_process_tree(proc_obj, timeout=5.0)
    except (psutil.NoSuchProcess, NameError):
        pass
