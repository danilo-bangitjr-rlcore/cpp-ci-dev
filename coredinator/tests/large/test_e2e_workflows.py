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

    assert wait_for_event(_agent_stopped, interval=0.1, timeout=3.0), "Agent should stop cleanly"


@pytest.mark.timeout(30)
def test_drayton_valley_workflow(
    coredinator_service: CoredinatorService,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test the Drayton Valley workflow with independent CoreIO service.

    This test demonstrates the Drayton Valley workflow where:
    1. CoreIO service is started independently via the CoreIO API
    2. Two agents (backwash and coag) connect to the existing CoreIO instance
    3. Agent status can be checked independently
    4. Agents can be stopped without affecting the independent CoreIO service
    5. The CoreIO service persists until explicitly stopped via the CoreIO API

    NOTE: This demonstrates independent CoreIO service management.
    For agent-managed service sharing, see test_agent_shared_coreio_service.
    """
    base_url = coredinator_service.base_url
    shared_coreio_id = "drayton-valley-coreio"

    # Create config files for backwash and coag agents
    backwash_config = config_file.parent / "backwash_config.yaml"
    backwash_config.write_text("agent_type: backwash\nprocess_params: {}\n")

    coag_config = config_file.parent / "coag_config.yaml"
    coag_config.write_text("agent_type: coag\nprocess_params: {}\n")

    # Step 1: Start CoreIO service independently via CoreIO API
    coreio_start_response = requests.post(
        f"{base_url}/api/io/start",
        json={
            "config_path": str(backwash_config),  # Use backwash config for CoreIO
            "coreio_id": shared_coreio_id,
        },
    )
    assert coreio_start_response.status_code == 200, f"Failed to start CoreIO: {coreio_start_response.text}"

    # Verify CoreIO is running independently
    coreio_status_response = requests.get(f"{base_url}/api/io/{shared_coreio_id}/status")
    assert coreio_status_response.status_code == 200
    coreio_status_data = coreio_status_response.json()
    assert coreio_status_data["status"]["state"] == "running"
    assert len(coreio_status_data["owners"]) == 1  # Only API owner

    # Step 2: Start backwash agent connecting to existing CoreIO service
    backwash_response = requests.post(
        f"{base_url}/api/agents/start",
        json={
            "config_path": str(backwash_config),
            "coreio_id": shared_coreio_id,
        },
    )
    assert backwash_response.status_code == 200, f"Failed to start backwash agent: {backwash_response.text}"
    backwash_id = backwash_response.json()

    # Step 3: Start coag agent connecting to the same existing CoreIO service
    coag_response = requests.post(
        f"{base_url}/api/agents/start",
        json={
            "config_path": str(coag_config),
            "coreio_id": shared_coreio_id,
        },
    )
    assert coag_response.status_code == 200, f"Failed to start coag agent: {coag_response.text}"
    coag_id = coag_response.json()

    # Wait for both agents to be running
    def _both_agents_running():
        backwash_status = requests.get(f"{base_url}/api/agents/{backwash_id}/status")
        coag_status = requests.get(f"{base_url}/api/agents/{coag_id}/status")
        return (
            backwash_status.status_code == 200 and backwash_status.json().get("state") == "running"
            and coag_status.status_code == 200 and coag_status.json().get("state") == "running"
        )

    assert wait_for_event(_both_agents_running, interval=0.1, timeout=5.0), "Both agents should be running"

    # Verify CoreIO now has multiple owners (API + agents)
    coreio_status_response = requests.get(f"{base_url}/api/io/{shared_coreio_id}/status")
    assert coreio_status_response.status_code == 200
    coreio_status_data = coreio_status_response.json()
    assert coreio_status_data["status"]["state"] == "running"
    assert len(coreio_status_data["owners"]) > 1  # API owner + agent owners
    assert coreio_status_data["is_shared"]

    # Verify both agents share the same CoreIO PID but have different CoreRL PIDs
    backwash_pids = get_microservice_pids(dist_with_fake_executable, backwash_id)
    coag_pids = get_microservice_pids(dist_with_fake_executable, coag_id)

    assert backwash_pids["coreio"] is not None, "Backwash agent should have CoreIO process"
    assert coag_pids["coreio"] is not None, "Coag agent should have CoreIO process"
    assert backwash_pids["coreio"] == coag_pids["coreio"], "Both agents should share the same CoreIO process"

    assert backwash_pids["corerl"] != coag_pids["corerl"], "Agents should have different CoreRL processes"

    # Check individual agent status
    backwash_status = requests.get(f"{base_url}/api/agents/{backwash_id}/status")
    assert backwash_status.status_code == 200
    assert backwash_status.json()["state"] == "running"
    assert backwash_status.json()["id"] == backwash_id

    coag_status = requests.get(f"{base_url}/api/agents/{coag_id}/status")
    assert coag_status.status_code == 200
    assert coag_status.json()["state"] == "running"
    assert coag_status.json()["id"] == coag_id

    # Step 4: Stop backwash agent - CoreIO should remain running (still owned by API and coag)
    stop_response = requests.post(f"{base_url}/api/agents/{backwash_id}/stop")
    assert stop_response.status_code == 200, f"Failed to stop backwash agent: {stop_response.text}"

    # Verify backwash is stopped but coag still running with same CoreIO PID
    def _backwash_stopped_coag_running():
        backwash_status = requests.get(f"{base_url}/api/agents/{backwash_id}/status")
        coag_status = requests.get(f"{base_url}/api/agents/{coag_id}/status")
        return (
            backwash_status.status_code == 200 and backwash_status.json().get("state") == "stopped"
            and coag_status.status_code == 200 and coag_status.json().get("state") == "running"
        )

    assert wait_for_event(_backwash_stopped_coag_running, interval=0.1, timeout=2.0), (
        "Backwash should be stopped while coag remains running"
    )

    # Verify CoreIO process is still the same and still has multiple owners
    coag_pids_after = get_microservice_pids(dist_with_fake_executable, coag_id)
    assert coag_pids_after["coreio"] == coag_pids["coreio"], (
        "Coag should still have the same CoreIO process after backwash stops"
    )

    coreio_status_response = requests.get(f"{base_url}/api/io/{shared_coreio_id}/status")
    assert coreio_status_response.status_code == 200
    coreio_status_data = coreio_status_response.json()
    assert coreio_status_data["status"]["state"] == "running"
    assert len(coreio_status_data["owners"]) >= 2  # Still API owner + coag agent

    # Step 5: Stop coag agent - CoreIO should still remain running (still owned by API)
    stop_response = requests.post(f"{base_url}/api/agents/{coag_id}/stop")
    assert stop_response.status_code == 200, f"Failed to stop coag agent: {stop_response.text}"

    # Verify both agents are stopped
    def _both_agents_stopped():
        backwash_status = requests.get(f"{base_url}/api/agents/{backwash_id}/status")
        coag_status = requests.get(f"{base_url}/api/agents/{coag_id}/status")
        return (
            backwash_status.status_code == 200 and backwash_status.json().get("state") == "stopped"
            and coag_status.status_code == 200 and coag_status.json().get("state") == "stopped"
        )

    assert wait_for_event(_both_agents_stopped, interval=0.1, timeout=2.0), (
        "Both agents should be stopped"
    )

    # Step 6: Verify CoreIO is still running independently (only API owner remains)
    coreio_status_response = requests.get(f"{base_url}/api/io/{shared_coreio_id}/status")
    assert coreio_status_response.status_code == 200
    coreio_status_data = coreio_status_response.json()
    assert coreio_status_data["status"]["state"] == "running"
    assert len(coreio_status_data["owners"]) == 1  # Only API owner remains
    assert not coreio_status_data["is_shared"]  # No longer shared

    # Step 7: Finally stop CoreIO via API
    coreio_stop_response = requests.post(f"{base_url}/api/io/{shared_coreio_id}/stop")
    assert coreio_stop_response.status_code == 200, f"Failed to stop CoreIO: {coreio_stop_response.text}"

    # Verify CoreIO is now stopped/removed
    coreio_status_response = requests.get(f"{base_url}/api/io/{shared_coreio_id}/status")
    assert coreio_status_response.status_code == 404, "CoreIO should be not found after stopping"
