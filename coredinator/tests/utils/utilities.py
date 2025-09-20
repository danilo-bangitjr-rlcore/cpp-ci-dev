from pathlib import Path
from typing import Any, Literal, Protocol

import requests

from coredinator.utils.test_polling import wait_for_event

# Type aliases for better code readability
AgentState = Literal["running", "stopped", "failed", "starting"]


# Type protocols for test client interfaces
class TestClient(Protocol):
    def get(self, url: str) -> Any: ...


class AgentManager(Protocol):
    def get_agent_status(self, agent_id: Any) -> Any: ...


class Service(Protocol):
    def status(self) -> Any: ...


# ============================================================================
# HTTP API Test Client
# ============================================================================

class CoredinatorAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def start_agent(self, config_path: str, coreio_id: str | None = None) -> str:
        payload = {"config_path": config_path}
        if coreio_id:
            payload["coreio_id"] = coreio_id

        response = requests.post(f"{self.base_url}/api/agents/start", json=payload)
        assert response.status_code == 200, f"Failed to start agent: {response.text}"
        return response.json()

    def stop_agent(self, agent_id: str) -> None:
        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/stop")
        assert response.status_code == 200, f"Failed to stop agent: {response.text}"

    def get_agent_status(self, agent_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/agents/{agent_id}/status")
        assert response.status_code == 200, f"Failed to get agent status: {response.text}"
        return response.json()

    def start_coreio_service(self, config_path: str, coreio_id: str | None = None) -> str:
        payload = {"config_path": config_path}
        if coreio_id:
            payload["coreio_id"] = coreio_id

        response = requests.post(f"{self.base_url}/api/io/start", json=payload)
        assert response.status_code == 200, f"Failed to start CoreIO: {response.text}"
        return response.json()["service_id"]

    def stop_coreio_service(self, service_id: str) -> None:
        response = requests.post(f"{self.base_url}/api/io/{service_id}/stop")
        assert response.status_code == 200, f"Failed to stop CoreIO: {response.text}"

    def get_coreio_status(self, service_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/io/{service_id}/status")
        assert response.status_code == 200, f"Failed to get CoreIO status: {response.text}"
        return response.json()

    def wait_for_agent_state(self, agent_id: str, expected_state: AgentState, timeout: float = 2.0) -> bool:
        return wait_for_agent_http_state(self.base_url, agent_id, expected_state, timeout)

    def assert_agent_state(self, agent_id: str, expected_state: AgentState, timeout: float = 2.0) -> None:
        assert_agent_http_state(self.base_url, agent_id, expected_state, timeout)


# ============================================================================
# Service Health and Process Verification Utilities
# ============================================================================

def get_service_process_ids(base_url: str, agent_id: str) -> dict[str, int | None]:
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
        service_info = service_statuses[corerl_id]
        if service_info.get("process_ids"):
            corerl_pid = service_info["process_ids"][0]

    if coreio_id in service_statuses:
        service_info = service_statuses[coreio_id]
        if service_info.get("process_ids"):
            coreio_pid = service_info["process_ids"][0]

    return {"corerl": corerl_pid, "coreio": coreio_pid}


def verify_service_statuses(agent_status: dict[str, Any], expected_services: list[str]) -> None:
    service_statuses = agent_status.get("service_statuses", {})

    for service in expected_services:
        assert service in service_statuses, f"Agent should have {service} service"
        assert service_statuses[service]["state"] == "running", f"{service} service should be running"


def verify_agents_independent(
    api_client: CoredinatorAPIClient,
    agent1_id: str,
    agent2_id: str,
) -> None:
    agent1_status = api_client.get_agent_status(agent1_id)
    agent2_status = api_client.get_agent_status(agent2_id)

    agent1_coreio_id = agent1_status["service_statuses"]["coreio"]["id"]
    agent2_coreio_id = agent2_status["service_statuses"]["coreio"]["id"]

    assert agent1_coreio_id != agent2_coreio_id, \
        f"Agents should have independent CoreIO services, but both have: {agent1_coreio_id}"


def verify_shared_service_access(
    api_client: CoredinatorAPIClient,
    agent_id: str,
    shared_service_name: str = "coreio",
) -> None:
    agent_status = api_client.get_agent_status(agent_id)
    agent_services = agent_status.get("service_statuses", {})

    assert shared_service_name in agent_services, \
        f"Agent {agent_id} should still have access to {shared_service_name} service"
    assert agent_services[shared_service_name]["state"] == "running", \
        f"Agent {agent_id}'s {shared_service_name} service should still be running"


# ============================================================================
# Config File Creation Utilities
# ============================================================================

def create_test_configs(base_path: Path, config_names: list[str]) -> dict[str, Path]:
    configs = {}
    for name in config_names:
        config_path = base_path / f"{name}_config.yaml"
        if name in ["backwash", "coag"]:
            config_path.write_text(f"agent_type: {name}\nprocess_params: {{}}\n")
        else:
            config_path.write_text("dummy: true\n")
        configs[name] = config_path
    return configs

def wait_for_agent_http_state(
    base_url: str,
    agent_id: str,
    expected_state: AgentState,
    timeout: float = 2.0,
    interval: float = 0.1,
) -> bool:
    def _check_agent_state():
        # Let HTTP and JSON errors propagate - they indicate real problems
        response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
        if response.status_code != 200:
            return False
        return response.json().get("state") == expected_state

    return wait_for_event(_check_agent_state, interval=interval, timeout=timeout)


def assert_agent_http_state(
    base_url: str,
    agent_id: str,
    expected_state: AgentState,
    timeout: float = 2.0,
) -> None:
    assert wait_for_agent_http_state(base_url, agent_id, expected_state, timeout), \
        f"Agent {agent_id} did not reach state '{expected_state}' within {timeout}s"


def get_agent_service_health(base_url: str, agent_id: str) -> dict[str, bool]:
    response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
    if response.status_code != 200:
        return {"corerl": False, "coreio": False}

    agent_status = response.json()
    service_statuses = agent_status.get("service_statuses", {})

    # Note: service_statuses uses short keys "corerl" and "coreio"
    corerl_healthy = service_statuses.get("corerl", {}).get("state") == "running"
    coreio_healthy = service_statuses.get("coreio", {}).get("state") == "running"

    return {"corerl": corerl_healthy, "coreio": coreio_healthy}


def verify_agent_services_running(base_url: str, agent_id: str) -> bool:
    health = get_agent_service_health(base_url, agent_id)
    return health["corerl"] and health["coreio"]


def assert_agent_services_healthy(base_url: str, agent_id: str) -> None:
    assert verify_agent_services_running(base_url, agent_id), \
        f"Agent {agent_id} services are not healthy"


def wait_for_agent_services_running(
    base_url: str,
    agent_id: str,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> bool:
    return wait_for_event(
        lambda: verify_agent_services_running(base_url, agent_id),
        interval=interval,
        timeout=timeout,
    )


# ============================================================================
# Service Sharing Verification Utilities
# ============================================================================

def verify_service_sharing(
    base_url: str,
    agent1_id: str,
    agent2_id: str,
) -> bool:
    response1 = requests.get(f"{base_url}/api/agents/{agent1_id}/status")
    response2 = requests.get(f"{base_url}/api/agents/{agent2_id}/status")

    if response1.status_code != 200 or response2.status_code != 200:
        return False

    agent1_status = response1.json()
    agent2_status = response2.json()

    # Both agents should be running
    if agent1_status.get("state") != "running" or agent2_status.get("state") != "running":
        return False

    # For service sharing tests, check if both agents have CoreIO services running
    service1_statuses = agent1_status.get("service_statuses", {})
    service2_statuses = agent2_status.get("service_statuses", {})

    # Both should have running CoreIO services (even if shared, they both access it)
    coreio1_running = service1_statuses.get("coreio", {}).get("state") == "running"
    coreio2_running = service2_statuses.get("coreio", {}).get("state") == "running"

    return coreio1_running and coreio2_running


# ============================================================================
# TestClient Agent State Verification Utilities
# ============================================================================

def wait_for_agent_testclient_state(
    client: TestClient,
    agent_id: str,
    expected_state: AgentState,
    timeout: float = 2.0,
    interval: float = 0.05,
) -> bool:
    def _check_agent_state():
        response = client.get(f"/api/agents/{agent_id}/status")
        if response.status_code != 200:
            return False
        return response.json().get("state") == expected_state

    return wait_for_event(_check_agent_state, interval=interval, timeout=timeout)


def assert_agent_testclient_state(
    client: TestClient,
    agent_id: str,
    expected_state: AgentState,
    timeout: float = 2.0,
) -> None:
    assert wait_for_agent_testclient_state(client, agent_id, expected_state, timeout), \
        f"Agent {agent_id} did not reach state '{expected_state}' within {timeout}s"


def wait_for_all_agents_testclient_state(
    client: TestClient,
    agent_ids: list[str],
    expected_state: AgentState,
    timeout: float = 5.0,
    interval: float = 0.05,
) -> bool:
    def _all_agents_in_state():
        for agent_id in agent_ids:
            response = client.get(f"/api/agents/{agent_id}/status")
            if response.status_code != 200:
                return False
            if response.json().get("state") != expected_state:
                return False
        return True

    return wait_for_event(_all_agents_in_state, interval=interval, timeout=timeout)


def assert_all_agents_testclient_state(
    client: TestClient,
    agent_ids: list[str],
    expected_state: AgentState,
    timeout: float = 5.0,
) -> None:
    assert wait_for_all_agents_testclient_state(client, agent_ids, expected_state, timeout), \
        f"Not all agents {agent_ids} reached state '{expected_state}' within {timeout}s"


# ============================================================================
# Direct Service/Manager State Verification Utilities
# ============================================================================

def wait_for_agent_manager_state(
    manager: AgentManager,
    agent_id: Any,
    expected_state: AgentState,
    timeout: float = 2.0,
    interval: float = 0.05,
) -> bool:
    def _check_manager_state():
        return manager.get_agent_status(agent_id).state == expected_state

    return wait_for_event(_check_manager_state, interval=interval, timeout=timeout)


def assert_agent_manager_state(
    manager: AgentManager,
    agent_id: Any,
    expected_state: AgentState,
    timeout: float = 2.0,
) -> None:
    assert wait_for_agent_manager_state(manager, agent_id, expected_state, timeout), \
        f"Agent {agent_id} did not reach state '{expected_state}' within {timeout}s via manager"


def wait_for_service_state(
    service: Service,
    expected_state: AgentState,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> bool:
    def _check_service_state():
        return service.status().state == expected_state

    return wait_for_event(_check_service_state, interval=interval, timeout=timeout)


def assert_service_state(
    service: Service,
    expected_state: AgentState,
    timeout: float = 5.0,
) -> None:
    assert wait_for_service_state(service, expected_state, timeout), \
        f"Service did not reach state '{expected_state}' within {timeout}s"


# ============================================================================
# Multi-Agent Test Utilities
# ============================================================================

def wait_for_all_agents_state(
    base_url: str,
    agent_ids: list[str],
    expected_state: AgentState,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> bool:
    def _all_agents_in_state():
        for agent_id in agent_ids:
            response = requests.get(f"{base_url}/api/agents/{agent_id}/status")
            if response.status_code != 200:
                return False
            if response.json().get("state") != expected_state:
                return False
        return True

    return wait_for_event(_all_agents_in_state, interval=interval, timeout=timeout)


def assert_all_agents_state(
    base_url: str,
    agent_ids: list[str],
    expected_state: AgentState,
    timeout: float = 5.0,
) -> None:
    assert wait_for_all_agents_state(base_url, agent_ids, expected_state, timeout), \
        f"Not all agents {agent_ids} reached state '{expected_state}' within {timeout}s"
