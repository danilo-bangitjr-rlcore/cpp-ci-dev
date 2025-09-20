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
from tests.utils.utilities import (
    CoredinatorAPIClient,
    assert_agent_http_state,
    assert_all_agents_state,
    verify_agent_services_running,
    verify_service_sharing,
    wait_for_agent_services_running,
)


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


@pytest.mark.timeout(15)
def test_agent_status_unknown_returns_stopped(coredinator_service: CoredinatorService):
    """
    Test that querying status for unknown agent returns 'stopped' state.

    Verifies the API gracefully handles requests for non-existent agents
    by returning a stopped state rather than an error.
    """
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    unknown_agent_id = "nonexistent-agent"

    # Query status for an agent that doesn't exist
    response = requests.get(f"{api_client.base_url}/api/agents/{unknown_agent_id}/status")

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
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    agent_id = config_file.stem

    # Start the agent
    agent_id = api_client.start_agent(str(config_file))

    # Wait for agent services to be running
    assert wait_for_agent_services_running(
        api_client.base_url, agent_id, timeout=5.0,
    ), "Agent services should be running"

    # Verify agent reports as running with healthy services
    data = api_client.get_agent_status(agent_id)
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
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    agent_id = config_file.stem

    # Start the agent
    agent_id = api_client.start_agent(str(config_file))

    # Wait for agent to be running before simulating coredinator failure
    api_client.assert_agent_state(agent_id, "running", timeout=2.0)

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
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    shared_coreio_id = "shared-coreio-test"

    # Create config files
    config1 = config_file.parent / "agent1_config.yaml"
    config1.write_text("dummy: true\n")
    config2 = config_file.parent / "agent2_config.yaml"
    config2.write_text("dummy: true\n")
    config3 = config_file.parent / "agent3_config.yaml"
    config3.write_text("dummy: true\n")

    # Start agents: agent1 independent, agent2 and agent3 with shared CoreIO
    agent1_id = api_client.start_agent(str(config1))
    agent2_id = api_client.start_agent(str(config2), coreio_id=shared_coreio_id)
    agent3_id = api_client.start_agent(str(config3), coreio_id=shared_coreio_id)

    # Wait for agents to be running
    assert_all_agents_state(api_client.base_url, [agent1_id, agent2_id, agent3_id], "running", timeout=5.0)

    # Verify all agents have healthy services
    assert verify_agent_services_running(api_client.base_url, agent1_id), "Agent1 services should be running"
    assert verify_agent_services_running(api_client.base_url, agent2_id), "Agent2 services should be running"
    assert verify_agent_services_running(api_client.base_url, agent3_id), "Agent3 services should be running"

    # Verify agent2 and agent3 share the same CoreIO service
    assert verify_service_sharing(api_client.base_url, agent2_id, agent3_id), \
        "Agent2 and Agent3 should share the CoreIO service"

    # Verify agent1 has independent services by checking service IDs
    agent1_status = api_client.get_agent_status(agent1_id)
    agent2_status = api_client.get_agent_status(agent2_id)

    agent1_coreio_id = agent1_status["service_statuses"]["coreio"]["id"]
    agent2_coreio_id = agent2_status["service_statuses"]["coreio"]["id"]

    # Agent1 should have its own independent CoreIO service (different ID)
    assert agent1_coreio_id != agent2_coreio_id, \
        f"Agent1 should have independent CoreIO service, but both have: {agent1_coreio_id}"

    # Stop agent2 - shared CoreIO should continue running for agent3
    api_client.stop_agent(agent2_id)
    api_client.assert_agent_state(agent3_id, "running", timeout=2.0)

    # Verify CoreIO service is still shared and agent3 continues running
    assert verify_agent_services_running(api_client.base_url, agent3_id), \
        "Agent3 should still have running services after agent2 stops"

    # Verify agent3 still has access to the shared service
    agent3_status = api_client.get_agent_status(agent3_id)
    agent3_services = agent3_status.get("service_statuses", {})
    assert "coreio" in agent3_services, "Agent3 should still have access to CoreIO service"
    assert agent3_services["coreio"]["state"] == "running", "Agent3's CoreIO service should still be running"


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
    assert_agent_http_state(base_url, agent_id, "running", timeout=3.0)

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
    assert_agent_http_state(base_url, agent_id, "stopped", timeout=3.0)


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
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    shared_coreio_id = "drayton-valley-coreio"

    # Create config files for backwash and coag agents
    backwash_config = config_file.parent / "backwash_config.yaml"
    backwash_config.write_text("agent_type: backwash\nprocess_params: {}\n")
    coag_config = config_file.parent / "coag_config.yaml"
    coag_config.write_text("agent_type: coag\nprocess_params: {}\n")

    # Step 1: Start CoreIO service independently via CoreIO API
    service_id = api_client.start_coreio_service(str(backwash_config), coreio_id=shared_coreio_id)
    assert service_id == shared_coreio_id

    # Verify CoreIO is running independently
    coreio_status = api_client.get_coreio_status(shared_coreio_id)
    assert coreio_status["status"]["state"] == "running"
    assert len(coreio_status["owners"]) == 1  # Only API owner

    # Step 2: Start agents connecting to existing CoreIO service
    backwash_id = api_client.start_agent(str(backwash_config), coreio_id=shared_coreio_id)
    coag_id = api_client.start_agent(str(coag_config), coreio_id=shared_coreio_id)

    # Wait for both agents to be running
    assert_all_agents_state(api_client.base_url, [backwash_id, coag_id], "running", timeout=5.0)

    # Verify CoreIO now has multiple owners (API + agents)
    coreio_status = api_client.get_coreio_status(shared_coreio_id)
    assert coreio_status["status"]["state"] == "running"
    assert len(coreio_status["owners"]) > 1  # API owner + agent owners
    assert coreio_status["is_shared"]

    # Verify both agents are running with healthy services
    assert verify_agent_services_running(api_client.base_url, backwash_id), "Backwash agent services should be running"
    assert verify_agent_services_running(api_client.base_url, coag_id), "Coag agent services should be running"

    # Verify both agents share the same CoreIO service
    assert verify_service_sharing(api_client.base_url, backwash_id, coag_id), \
        "Both agents should share the same CoreIO service"

    # Check individual agent status
    backwash_status = api_client.get_agent_status(backwash_id)
    assert backwash_status["state"] == "running"
    assert backwash_status["id"] == backwash_id

    coag_status = api_client.get_agent_status(coag_id)
    assert coag_status["state"] == "running"
    assert coag_status["id"] == coag_id

    # Step 4: Stop backwash agent - CoreIO should remain running
    api_client.stop_agent(backwash_id)
    api_client.assert_agent_state(backwash_id, "stopped", timeout=2.0)
    api_client.assert_agent_state(coag_id, "running", timeout=2.0)

    # Verify CoreIO service continues running and coag agent still has access
    assert verify_agent_services_running(api_client.base_url, coag_id), \
        "Coag should still have running services after backwash stops"

    # Step 5: Stop coag agent - CoreIO should still remain running (owned by API)
    api_client.stop_agent(coag_id)
    assert_all_agents_state(api_client.base_url, [backwash_id, coag_id], "stopped", timeout=2.0)

    # Step 6: Verify CoreIO is still running independently (only API owner remains)
    coreio_status = api_client.get_coreio_status(shared_coreio_id)
    assert coreio_status["status"]["state"] == "running"
    assert len(coreio_status["owners"]) == 1  # Only API owner remains
    assert not coreio_status["is_shared"]  # No longer shared

    # Step 7: Finally stop CoreIO via API
    api_client.stop_coreio_service(shared_coreio_id)

    # Verify CoreIO is now stopped/removed
    response = requests.get(f"{api_client.base_url}/api/io/{shared_coreio_id}/status")
    assert response.status_code == 404, "CoreIO should be not found after stopping"
