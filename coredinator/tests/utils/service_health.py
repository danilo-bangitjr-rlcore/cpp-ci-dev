"""Service health and process verification utilities for coredinator testing."""

from typing import TYPE_CHECKING, Any

import requests

from .polling import wait_for_event

if TYPE_CHECKING:
    from .api_client import CoredinatorAPIClient


def get_service_process_ids(base_url: str, agent_id: str) -> dict[str, int | None]:
    """Get process IDs for agent services."""
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
    """Verify that agent has expected services running."""
    service_statuses = agent_status.get("service_statuses", {})

    for service in expected_services:
        assert service in service_statuses, (
            f"Agent should have {service} service; available: {list(service_statuses.keys())}"
        )
        state = service_statuses[service].get("state")
        assert state == "running", (
            f"{service} service should be running; actual state: {state!r}"
        )


def verify_agents_independent(
    api_client: "CoredinatorAPIClient",
    agent1_id: str,
    agent2_id: str,
) -> None:
    """Verify that two agents have independent CoreIO services."""
    agent1_status = api_client.get_agent_status(agent1_id)
    agent2_status = api_client.get_agent_status(agent2_id)

    agent1_coreio_id = agent1_status["service_statuses"]["coreio"]["id"]
    agent2_coreio_id = agent2_status["service_statuses"]["coreio"]["id"]

    assert agent1_coreio_id != agent2_coreio_id, (
        f"Agents should have independent CoreIO services, but both have: {agent1_coreio_id}"
    )


def verify_shared_service_access(
    api_client: "CoredinatorAPIClient",
    agent_id: str,
    shared_service_name: str = "coreio",
) -> None:
    """Verify that agent has access to a shared service."""
    agent_status = api_client.get_agent_status(agent_id)
    agent_services = agent_status.get("service_statuses", {})

    assert shared_service_name in agent_services, (
        f"Agent {agent_id} should still have access to {shared_service_name} service; "
        f"available: {list(agent_services.keys())}"
    )
    state = agent_services[shared_service_name].get("state")
    assert state == "running", (
        f"Agent {agent_id}'s {shared_service_name} service should still be running; actual: {state!r}"
    )


def get_agent_service_health(base_url: str, agent_id: str) -> dict[str, bool]:
    """Get health status of agent services."""
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
    """Verify that all agent services are running."""
    health = get_agent_service_health(base_url, agent_id)
    return health["corerl"] and health["coreio"]


def assert_agent_services_healthy(base_url: str, agent_id: str) -> None:
    """Assert that agent services are healthy."""
    healthy = verify_agent_services_running(base_url, agent_id)
    assert healthy, f"Agent {agent_id} services are not healthy"


def wait_for_agent_services_running(
    base_url: str,
    agent_id: str,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> bool:
    """Wait for agent services to be running."""
    return wait_for_event(
        lambda: verify_agent_services_running(base_url, agent_id),
        interval=interval,
        timeout=timeout,
    )


def verify_service_sharing(
    base_url: str,
    agent1_id: str,
    agent2_id: str,
) -> bool:
    """Verify that two agents are sharing services properly."""
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
