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
    """
    Test that querying status for unknown agent returns 'stopped' state.

    Verifies the API gracefully handles requests for non-existent agents
    by returning a stopped state rather than an error.
    """
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
    """
    Simulate microservice failure and verify coredinator recovers agent state.

    Tests the system's ability to detect when individual microservices fail
    and correctly report the agent state as failed.
    """
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
    """
    Simulate coredinator failure and verify agent state is restored after restart.

    Tests the persistence and recovery capabilities by killing the coredinator
    process and verifying agents can be tracked after service restart.
    """
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


@pytest.mark.timeout(30)
def test_agent_shared_coreio_service(
    coredinator_service: CoredinatorService,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that multiple agents can share a CoreIO service instance when coreio_id is provided.

    This test validates the new service sharing feature by:
    1. Starting agent1 normally (independent services)
    2. Starting agent2 and agent3 with shared CoreIO service
    3. Verifying agent2/agent3 share same CoreIO PID but have different CoreRL PIDs
    4. Stopping agent2 and confirming agent3 continues with shared CoreIO
    5. Ensuring agent1 remains completely independent throughout
    """
    base_url = coredinator_service.base_url
    shared_coreio_id = "shared-coreio-test"

    # Create first config file
    config1 = config_file.parent / "agent1_config.yaml"
    config1.write_text("dummy: true\n")

    # Create second config file
    config2 = config_file.parent / "agent2_config.yaml"
    config2.write_text("dummy: true\n")

    # Start first agent without shared service (normal behavior)
    response1 = requests.post(f"{base_url}/api/agents/start", json={
        "config_path": str(config1),
    })
    assert response1.status_code == 200, f"Failed to start agent1: {response1.text}"
    agent1_id = response1.json()

    # Start second agent with shared CoreIO service
    response2 = requests.post(f"{base_url}/api/agents/start", json={
        "config_path": str(config2),
        "coreio_id": shared_coreio_id,
    })
    assert response2.status_code == 200, f"Failed to start agent2: {response2.text}"
    agent2_id = response2.json()

    # Start third agent sharing the same CoreIO service
    config3 = config_file.parent / "agent3_config.yaml"
    config3.write_text("dummy: true\n")

    response3 = requests.post(f"{base_url}/api/agents/start", json={
        "config_path": str(config3),
        "coreio_id": shared_coreio_id,
    })
    assert response3.status_code == 200, f"Failed to start agent3: {response3.text}"
    agent3_id = response3.json()

    # Wait for agents to be running
    def _all_agents_running():
        for agent_id in [agent1_id, agent2_id, agent3_id]:
            response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
            if response.status_code != 200 or response.json().get("state") != "running":
                return False
        return True

    assert wait_for_event(_all_agents_running, interval=0.1, timeout=5.0), \
        "All agents should be running"

    # Verify distinct PIDs for agent1 (not sharing)
    pids1 = get_microservice_pids(dist_with_fake_executable, agent1_id)
    assert pids1["coreio"] is not None, "Agent1 should have its own CoreIO process"
    assert pids1["corerl"] is not None, "Agent1 should have its own CoreRL process"

    # Verify agent2 and agent3 share the same CoreIO process but have different CoreRL processes
    pids2 = get_microservice_pids(dist_with_fake_executable, agent2_id)
    pids3 = get_microservice_pids(dist_with_fake_executable, agent3_id)

    assert pids2["coreio"] is not None, "Agent2 should have CoreIO process"
    assert pids3["coreio"] is not None, "Agent3 should have CoreIO process"
    assert pids2["coreio"] == pids3["coreio"], "Agent2 and Agent3 should share the same CoreIO process"

    assert pids2["corerl"] != pids3["corerl"], "Agent2 and Agent3 should have different CoreRL processes"
    assert pids1["coreio"] != pids2["coreio"], "Agent1 should not share CoreIO with agent2/3"

    # Stop agent2 - shared CoreIO should continue running for agent3
    response = requests.post(f"{base_url}/api/agents/{agent2_id}/stop")
    assert response.status_code == 200, f"Failed to stop agent2: {response.text}"

    # Verify agent3 still running with same CoreIO PID
    def _agent3_still_running():
        response = requests.get(f"{base_url}/api/agents/{agent3_id}/status")
        return response.status_code == 200 and response.json().get("state") == "running"

    assert wait_for_event(_agent3_still_running, interval=0.1, timeout=2.0), \
        "Agent3 should still be running after agent2 stops"

    # Verify CoreIO process is still the same
    pids3_after = get_microservice_pids(dist_with_fake_executable, agent3_id)
    assert pids3_after["coreio"] == pids3["coreio"], \
        "Agent3 should still have the same CoreIO process after agent2 stops"


@pytest.mark.timeout(20)
def test_agent_start_backward_compatibility(
    coredinator_service: CoredinatorService,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that agents started without coreio_id behave exactly as before.

    Ensures the new optional coreio_id parameter doesn't break existing
    functionality by testing the traditional API usage patterns.
    """
    base_url = coredinator_service.base_url
    agent_id = config_file.stem

    # Start agent using the old API format (no coreio_id)
    response = requests.post(f"{base_url}/api/agents/start", json={
        "config_path": str(config_file),
    })
    assert response.status_code == 200, f"Failed to start agent: {response.text}"
    returned_id = response.json()
    assert returned_id == agent_id

    # Wait for agent to be running
    def _agent_running():
        response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
        return response.status_code == 200 and response.json().get("state") == "running"

    assert wait_for_event(_agent_running, interval=0.1, timeout=3.0), \
        "Agent should be running"

    # Verify agent has its own unique service instances
    pids = get_microservice_pids(dist_with_fake_executable, agent_id)
    assert pids["coreio"] is not None, "Agent should have CoreIO process"
    assert pids["corerl"] is not None, "Agent should have CoreRL process"

    # Verify agent status reports correctly
    response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == agent_id
    assert data["state"] == "running"
    assert data["config_path"] == str(config_file)

    # Verify agent can be stopped normally
    response = requests.post(f"{base_url}/api/agents/{agent_id}/stop")
    assert response.status_code == 200

    # Verify agent stops properly
    def _agent_stopped():
        response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
        return response.status_code == 200 and response.json().get("state") == "stopped"

    assert wait_for_event(_agent_stopped, interval=0.1, timeout=3.0), \
        "Agent should stop cleanly"
