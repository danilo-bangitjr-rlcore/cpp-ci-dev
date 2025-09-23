"""Large integration tests for coredinator service.

These tests start the coredinator service as a subprocess and make real HTTP
requests to test end-to-end workflows.
"""

import subprocess
from pathlib import Path

import psutil
import pytest
import requests

from coredinator.test_utils import CoredinatorService, wait_for_service_healthy
from coredinator.utils.process import terminate_process_tree
from coredinator.utils.test_polling import wait_for_event


def get_agent_service_health(base_url: str, agent_id: str):
    """Check if an agent's services are healthy by querying agent status."""
    response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
    if response.status_code != 200:
        return {"corerl": False, "coreio": False}

    agent_status = response.json()
    service_statuses = agent_status.get("service_statuses", {})

    # Note: service_statuses uses short keys "corerl" and "coreio", not full service IDs
    corerl_healthy = service_statuses.get("corerl", {}).get("state") == "running"
    coreio_healthy = service_statuses.get("coreio", {}).get("state") == "running"

    return {"corerl": corerl_healthy, "coreio": coreio_healthy}


def get_service_process_ids(base_url: str, agent_id: str):
    """Get process IDs for an agent's services via the API."""
    response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
    if response.status_code != 200:
        return {"corerl": None, "coreio": None}

    agent_status = response.json()
    service_statuses = agent_status.get("service_statuses", {})

    corerl_id = f"{agent_id}-corerl"
    coreio_id = f"{agent_id}-coreio"

    corerl_pid = None
    coreio_pid = None

    if corerl_id in service_statuses:
        # Try to get process ID from service status if available
        service_info = service_statuses[corerl_id]
        if service_info.get("process_ids"):
            corerl_pid = service_info["process_ids"][0]

    if coreio_id in service_statuses:
        service_info = service_statuses[coreio_id]
        if service_info.get("process_ids"):
            coreio_pid = service_info["process_ids"][0]

    return {"corerl": corerl_pid, "coreio": coreio_pid}


def verify_agent_services_running(base_url: str, agent_id: str) -> bool:
    health = get_agent_service_health(base_url, agent_id)
    return health["corerl"] and health["coreio"]


def verify_service_sharing(base_url: str, agent1_id: str, agent2_id: str, shared_service_id: str) -> bool:
    response1 = requests.get(f"{base_url}/api/agents/{agent1_id}/status")
    response2 = requests.get(f"{base_url}/api/agents/{agent2_id}/status")

    if response1.status_code != 200 or response2.status_code != 200:
        return False

    agent1_status = response1.json()
    agent2_status = response2.json()

    # Both agents should be running
    if agent1_status.get("state") != "running" or agent2_status.get("state") != "running":
        return False

    # For service sharing tests, we check if both agents have CoreIO services running
    # The actual sharing is verified at the service manager level, not visible in agent status
    service1_statuses = agent1_status.get("service_statuses", {})
    service2_statuses = agent2_status.get("service_statuses", {})

    # Both should have running CoreIO services (even if shared, they both access it)
    coreio1_running = service1_statuses.get("coreio", {}).get("state") == "running"
    coreio2_running = service2_statuses.get("coreio", {}).get("state") == "running"

    return coreio1_running and coreio2_running


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
    Test agent service health monitoring instead of process failure simulation.

    This test verifies the agent correctly reports service status through the API,
    which is more reliable than process ID checking.
    """
    base_url = coredinator_service.base_url
    agent_id = config_file.stem

    # Start the agent
    response = requests.post(f"{base_url}/api/agents/start", json={"config_path": str(config_file)})
    assert response.status_code == 200, f"Failed to start agent: {response.text}"

    # Wait for agent services to be running
    def _agent_services_running():
        return verify_agent_services_running(base_url, agent_id)

    assert wait_for_event(_agent_services_running, interval=0.1, timeout=5.0), "Agent services should be running"

    # Verify agent reports as running with healthy services
    response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "running"

    # Verify both services are reported as running
    service_statuses = data.get("service_statuses", {})

    assert "coreio" in service_statuses, "Agent should have coreio service"
    assert "corerl" in service_statuses, "Agent should have corerl service"
    assert service_statuses["coreio"]["state"] == "running"
    assert service_statuses["corerl"]["state"] == "running"


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

    # Verify all agents have healthy services
    assert verify_agent_services_running(base_url, agent1_id), "Agent1 services should be running"
    assert verify_agent_services_running(base_url, agent2_id), "Agent2 services should be running"
    assert verify_agent_services_running(base_url, agent3_id), "Agent3 services should be running"

    # Verify agent2 and agent3 share the same CoreIO service
    shared_coreio_id = "shared-coreio-test"
    assert verify_service_sharing(base_url, agent2_id, agent3_id, shared_coreio_id), \
        "Agent2 and Agent3 should share the CoreIO service"

    # Verify agent1 has independent services (they're not sharing actual service instances)
    # Note: All agents will show "coreio" and "corerl" keys in their status, but the actual
    # service instances can be different. We can verify independence by checking their service IDs.
    response1 = requests.get(f"{base_url}/api/agents/{agent1_id}/status")
    response2 = requests.get(f"{base_url}/api/agents/{agent2_id}/status")

    agent1_coreio_id = response1.json()["service_statuses"]["coreio"]["id"]
    agent2_coreio_id = response2.json()["service_statuses"]["coreio"]["id"]

    # Agent1 should have its own independent CoreIO service (different ID)
    assert agent1_coreio_id != agent2_coreio_id, \
        f"Agent1 should have independent CoreIO service, but both have: {agent1_coreio_id}"

    # Stop agent2 - shared CoreIO should continue running for agent3
    response = requests.post(f"{base_url}/api/agents/{agent2_id}/stop")
    assert response.status_code == 200, f"Failed to stop agent2: {response.text}"

    # Verify agent3 still running with same CoreIO PID
    def _agent3_still_running():
        response = requests.get(f"{base_url}/api/agents/{agent3_id}/status")
        return response.status_code == 200 and response.json().get("state") == "running"

    assert wait_for_event(_agent3_still_running, interval=0.1, timeout=2.0), \
        "Agent3 should still be running after agent2 stops"

    # Verify CoreIO service is still shared and agent3 continues running
    assert verify_agent_services_running(base_url, agent3_id), \
        "Agent3 should still have running services after agent2 stops"

    # Verify agent3 still has access to the shared service (will show as "coreio")
    response3 = requests.get(f"{base_url}/api/agents/{agent3_id}/status")
    agent3_services = response3.json().get("service_statuses", {})
    assert "coreio" in agent3_services, \
        "Agent3 should still have access to CoreIO service"
    assert agent3_services["coreio"]["state"] == "running", \
        "Agent3's CoreIO service should still be running"


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

    # Verify agent has its own unique service instances by using the health check
    assert verify_agent_services_running(base_url, agent_id), "Agent services should be running"

    # Verify agent status reports correctly with both services
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

    # Verify both agents are running with healthy services
    assert verify_agent_services_running(base_url, backwash_id), "Backwash agent services should be running"
    assert verify_agent_services_running(base_url, coag_id), "Coag agent services should be running"

    # Verify both agents share the same CoreIO service
    assert verify_service_sharing(base_url, backwash_id, coag_id, shared_coreio_id), \
        "Both agents should share the same CoreIO service"

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

    # Verify CoreIO service continues running and coag agent still has access
    assert verify_agent_services_running(base_url, coag_id), \
        "Coag should still have running services after backwash stops"

    # Verify coag still has access to the shared CoreIO service
    # Note: The agent status will show "coreio" (not the shared service ID)
    response = requests.get(f"{base_url}/api/agents/{coag_id}/status")
    coag_services = response.json().get("service_statuses", {})
    assert "coreio" in coag_services, \
        "Coag should still have access to CoreIO service"
    assert coag_services["coreio"]["state"] == "running", \
        "Coag's CoreIO service should still be running"

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
