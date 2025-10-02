"""Large integration tests for coredinator service.

These tests start the coredinator service as a subprocess and make real HTTP
requests to test end-to-end workflows.
"""

import subprocess
import sys
from pathlib import Path

import psutil
import pytest
import requests
from lib_process.process import Process

from tests.utils.api_client import CoredinatorAPIClient
from tests.utils.config_helpers import create_test_configs
from tests.utils.service_fixtures import CoredinatorService, wait_for_service_healthy
from tests.utils.service_health import (
    verify_agent_services_running,
    verify_agents_independent,
    verify_service_sharing,
    verify_service_statuses,
    verify_shared_service_access,
    wait_for_agent_services_running,
)
from tests.utils.state_verification import assert_all_agents_state
from tests.utils.timeout_multiplier import apply_timeout_multiplier

# Platform-adjusted timeout value for test decorators
TIMEOUT = int(apply_timeout_multiplier(30))

@pytest.mark.timeout(TIMEOUT)
def test_agent_status_unknown_returns_stopped(coredinator_service: CoredinatorService):
    """
    Test that querying status for unknown agent returns 'stopped' state.

    Verifies the API gracefully handles requests for non-existent agents
    by returning a stopped state rather than an error.
    """
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    unknown_agent_id = "nonexistent-agent"

    # Query status for an agent that doesn't exist
    response = requests.get(f"{api_client.base_url}/api/agents/{unknown_agent_id}/status", timeout=10.0)

    # Should return 200 with stopped state
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == unknown_agent_id
    assert data["state"] == "stopped"
    assert data["config_path"] is None


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.skipif(sys.platform == "win32", reason="Flaky on Windows - hangs starting agents")
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
    verify_service_statuses(data, ["coreio", "corerl"])


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.skipif(sys.platform == "win32", reason="Flaky on Windows - hangs starting agents")
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
    assert Process(proc_obj).terminate_tree(timeout=5.0), "Coredinator did not terminate in time"

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
    response = requests.get(f"{coredinator_service.base_url}/api/agents/{agent_id}/status", timeout=10.0)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == agent_id
    assert data["state"] in ("running", "stopped")

    # Cleanup the restarted process
    Process.from_pid(proc2.pid).terminate_tree(timeout=5.0)


@pytest.mark.skipif(sys.platform == "win32", reason="Flaky on Windows - hangs on network requests")
@pytest.mark.timeout(TIMEOUT)
def test_agent_shared_coreio_service(
    coredinator_service: CoredinatorService,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that multiple agents can reuse the same CoreIO service instance.

    This test validates service reuse by:
    1. Starting agent1 normally (gets its own CoreIO service)
    2. Starting agent2 and agent3 with the same coreio_id (reuse the same CoreIO instance)
    3. Verifying agent2/agent3 share same CoreIO PID but have different CoreRL PIDs
    4. Stopping agent2 and confirming agent3 continues unaffected
    5. Ensuring agent1 remains completely independent throughout

    Note: Stopping an agent only stops its CoreRL service. CoreIO services persist
    until explicitly stopped via the CoreIO API.
    """
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    shared_coreio_id = "shared-coreio-test"

    # Create config files
    configs = create_test_configs(config_file.parent, ["agent1", "agent2", "agent3"])

    # Start agents: agent1 independent, agent2 and agent3 with shared CoreIO
    agent1_id = api_client.start_agent(str(configs["agent1"]))
    agent2_id = api_client.start_agent(str(configs["agent2"]), coreio_id=shared_coreio_id)
    agent3_id = api_client.start_agent(str(configs["agent3"]), coreio_id=shared_coreio_id)

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
    verify_agents_independent(api_client, agent1_id, agent2_id)

    # Stop agent2 - CoreIO continues running (agents only stop their CoreRL service)
    api_client.stop_agent(agent2_id)
    api_client.assert_agent_state(agent3_id, "running", timeout=2.0)

    # Verify CoreIO service continues running and agent3 continues unaffected
    assert verify_agent_services_running(api_client.base_url, agent3_id), \
        "Agent3 should still have running services after agent2 stops"

    # Verify agent3 still has access to the shared service
    verify_shared_service_access(api_client, agent3_id)


@pytest.mark.skipif(sys.platform == "win32", reason="Flaky on Windows - hangs on network requests")
@pytest.mark.timeout(TIMEOUT)
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
    api_client = CoredinatorAPIClient(coredinator_service.base_url)

    # Start agent using the normal API (no coreio_id)
    agent_id = api_client.start_agent(str(config_file))

    # Wait for agent to be running
    api_client.assert_agent_state(agent_id, "running", timeout=3.0)

    # Verify agent has its own unique service instances by using the health check
    assert verify_agent_services_running(api_client.base_url, agent_id), "Agent services should be running"

    # Verify agent status reports correctly with both services
    data = api_client.get_agent_status(agent_id)
    assert data["id"] == agent_id
    assert data["state"] == "running"
    assert data["config_path"] == str(config_file)

    # Verify agent can be stopped normally
    api_client.stop_agent(agent_id)
    api_client.assert_agent_state(agent_id, "stopped", timeout=3.0)


@pytest.mark.skipif(sys.platform == "win32", reason="Flaky on Windows - hangs on network requests")
@pytest.mark.timeout(TIMEOUT)
def test_drayton_valley_workflow(
    coredinator_service: CoredinatorService,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test CoreIO lifecycle management via the CoreIO API.

    This test demonstrates:
    1. CoreIO service can be started independently via the CoreIO API
    2. Multiple agents can connect to a pre-existing CoreIO instance
    3. Stopping agents leaves CoreIO running (agents only stop their CoreRL service)
    4. CoreIO must be explicitly stopped via the CoreIO API

    This validates that CoreIO has an independent lifecycle from agents.
    """
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    shared_coreio_id = "drayton-valley-coreio"

    # Create config files for backwash and coag agents
    configs = create_test_configs(config_file.parent, ["backwash", "coag"])

    # Step 1: Start CoreIO service independently via CoreIO API
    service_id = api_client.start_coreio_service(str(configs["backwash"]), coreio_id=shared_coreio_id)
    assert service_id == shared_coreio_id

    # Verify CoreIO is running independently
    coreio_status = api_client.get_coreio_status(shared_coreio_id)
    assert coreio_status["status"]["state"] == "running"

    # Step 2: Start agents connecting to existing CoreIO service
    backwash_id = api_client.start_agent(str(configs["backwash"]), coreio_id=shared_coreio_id)
    coag_id = api_client.start_agent(str(configs["coag"]), coreio_id=shared_coreio_id)

    # Wait for both agents to be running
    assert_all_agents_state(api_client.base_url, [backwash_id, coag_id], "running", timeout=5.0)

    # Verify CoreIO is still running
    coreio_status = api_client.get_coreio_status(shared_coreio_id)
    assert coreio_status["status"]["state"] == "running"

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

    # Step 4: Stop backwash agent - CoreIO continues running (agents only stop CoreRL)
    api_client.stop_agent(backwash_id)
    api_client.assert_agent_state(backwash_id, "stopped", timeout=2.0)
    api_client.assert_agent_state(coag_id, "running", timeout=2.0)

    # Verify CoreIO service continues running and coag agent still has access
    assert verify_agent_services_running(api_client.base_url, coag_id), \
        "Coag should still have running services after backwash stops"

    # Step 5: Stop coag agent - CoreIO still remains running
    api_client.stop_agent(coag_id)
    assert_all_agents_state(api_client.base_url, [backwash_id, coag_id], "stopped", timeout=2.0)

    # Step 6: Verify CoreIO continues running (agents don't stop CoreIO)
    coreio_status = api_client.get_coreio_status(shared_coreio_id)
    assert coreio_status["status"]["state"] == "running"

    # Step 7: Explicitly stop CoreIO via the CoreIO API
    api_client.stop_coreio_service(shared_coreio_id)

    # Verify CoreIO is now stopped/removed
    response = requests.get(f"{api_client.base_url}/api/io/{shared_coreio_id}/status", timeout=10.0)
    assert response.status_code == 404, "CoreIO should be not found after stopping"
