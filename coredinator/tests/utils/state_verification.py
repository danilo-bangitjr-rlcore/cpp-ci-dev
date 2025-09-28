"""Agent and service state verification utilities for coredinator testing."""

from typing import Any

import requests

from .polling import wait_for_event
from .types import AgentManager, AgentState, Service, TestClient

# ============================================================================
# HTTP-based State Verification
# ============================================================================

def wait_for_agent_http_state(
    base_url: str,
    agent_id: str,
    expected_state: AgentState,
    timeout: float = 2.0,
    interval: float = 0.1,
) -> bool:
    """Wait for agent to reach expected state via HTTP API."""
    def _check_agent_state():
        # Let HTTP and JSON errors propagate - they indicate real problems
        response = requests.get(f"{base_url}/api/agents/{agent_id}/status", timeout=10.0)
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
    """Assert that agent reaches expected state via HTTP API."""
    last_state: str | None = None

    def _check_agent_state():
        nonlocal last_state
        try:
            response = requests.get(f"{base_url}/api/agents/{agent_id}/status", timeout=10.0)
        except Exception as e:
            last_state = f"error: {e}"
            return False

        if response.status_code != 200:
            last_state = f"http:{response.status_code}"
            return False

        last_state = response.json().get("state")
        return last_state == expected_state

    ok = wait_for_event(_check_agent_state, interval=0.1, timeout=timeout)
    assert ok, (
        f"Agent {agent_id} did not reach expected state {expected_state!r} within {timeout}s "
        f"(checked {base_url}/api/agents/{agent_id}/status). Last observed state: {last_state!r}"
    )


def wait_for_all_agents_state(
    base_url: str,
    agent_ids: list[str],
    expected_state: AgentState,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> bool:
    """Wait for all agents to reach expected state via HTTP API."""
    def _all_agents_in_state():
        for agent_id in agent_ids:
            response = requests.get(f"{base_url}/api/agents/{agent_id}/status", timeout=10.0)
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
    """Assert that all agents reach expected state via HTTP API."""
    last_states: dict[str, str | None] = dict.fromkeys(agent_ids)

    def _all_agents_in_state():
        for aid in agent_ids:
            try:
                response = requests.get(f"{base_url}/api/agents/{aid}/status", timeout=10.0)
            except Exception as e:
                last_states[aid] = f"error: {e}"
                return False

            if response.status_code != 200:
                last_states[aid] = f"http:{response.status_code}"
                return False

            state = response.json().get("state")
            last_states[aid] = state
            if state != expected_state:
                return False
        return True

    ok = wait_for_event(_all_agents_in_state, interval=0.1, timeout=timeout)
    if not ok:
        raise AssertionError(
            f"Not all agents {agent_ids} reached expected state {expected_state!r} within {timeout}s "
            f"(checked via {base_url}/api/agents/<id>/status). Last observed states: {last_states}",
        )


# ============================================================================
# TestClient-based State Verification
# ============================================================================

def wait_for_agent_testclient_state(
    client: TestClient,
    agent_id: str,
    expected_state: AgentState,
    timeout: float = 2.0,
    interval: float = 0.05,
) -> bool:
    """Wait for agent to reach expected state via TestClient."""
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
    """Assert that agent reaches expected state via TestClient."""
    last_state: str | None = None

    def _check_agent_state():
        nonlocal last_state
        try:
            response = client.get(f"/api/agents/{agent_id}/status")
        except Exception as e:
            last_state = f"error: {e}"
            return False

        if response.status_code != 200:
            last_state = f"http:{response.status_code}"
            return False

        last_state = response.json().get("state")
        return last_state == expected_state

    ok = wait_for_event(_check_agent_state, interval=0.05, timeout=timeout)
    assert ok, (
        f"Agent {agent_id} did not reach expected state {expected_state!r} within {timeout}s "
        f"(GET /api/agents/{agent_id}/status via TestClient={client!r}). Last observed state: {last_state!r}"
    )


def wait_for_all_agents_testclient_state(
    client: TestClient,
    agent_ids: list[str],
    expected_state: AgentState,
    timeout: float = 5.0,
    interval: float = 0.05,
) -> bool:
    """Wait for all agents to reach expected state via TestClient."""
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
    """Assert that all agents reach expected state via TestClient."""
    last_states: dict[str, str | None] = dict.fromkeys(agent_ids)

    def _all_agents_in_state():
        for aid in agent_ids:
            try:
                response = client.get(f"/api/agents/{aid}/status")
            except Exception as e:
                last_states[aid] = f"error: {e}"
                return False

            if response.status_code != 200:
                last_states[aid] = f"http:{response.status_code}"
                return False

            state = response.json().get("state")
            last_states[aid] = state
            if state != expected_state:
                return False
        return True

    ok = wait_for_event(_all_agents_in_state, interval=0.05, timeout=timeout)
    if not ok:
        raise AssertionError(
            f"Not all agents {agent_ids} reached expected state {expected_state!r} within {timeout}s "
            f"(GET /api/agents/<id>/status via TestClient={client!r}). Last observed states: {last_states}",
        )


# ============================================================================
# Direct Manager/Service State Verification
# ============================================================================

def wait_for_agent_manager_state(
    manager: AgentManager,
    agent_id: Any,
    expected_state: AgentState,
    timeout: float = 2.0,
    interval: float = 0.05,
) -> bool:
    """Wait for agent to reach expected state via direct manager access."""
    def _check_manager_state():
        return manager.get_agent_status(agent_id).state == expected_state

    return wait_for_event(_check_manager_state, interval=interval, timeout=timeout)


def assert_agent_manager_state(
    manager: AgentManager,
    agent_id: Any,
    expected_state: AgentState,
    timeout: float = 2.0,
) -> None:
    """Assert that agent reaches expected state via direct manager access."""
    last_state: str | None = None

    def _check_manager_state():
        nonlocal last_state
        try:
            status = manager.get_agent_status(agent_id)
            last_state = getattr(status, "state", None)
        except Exception as e:
            last_state = f"error: {e}"
            return False

        return last_state == expected_state

    ok = wait_for_event(_check_manager_state, interval=0.05, timeout=timeout)
    assert ok, (
        f"Agent {agent_id} did not reach expected state {expected_state!r} within {timeout}s via manager "
        f"(manager={manager!r}). Last observed state: {last_state!r}"
    )


def wait_for_service_state(
    service: Service,
    expected_state: AgentState,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> bool:
    """Wait for service to reach expected state via direct service access."""
    def _check_service_state():
        return service.status().state == expected_state

    return wait_for_event(_check_service_state, interval=interval, timeout=timeout)


def assert_service_state(
    service: Service,
    expected_state: AgentState,
    timeout: float = 5.0,
) -> None:
    """Assert that service reaches expected state via direct service access."""
    last_state: str | None = None

    def _check_service_state():
        nonlocal last_state
        try:
            status = service.status()
            last_state = getattr(status, "state", None)
        except Exception as e:
            last_state = f"error: {e}"
            return False

        return last_state == expected_state

    ok = wait_for_event(_check_service_state, interval=0.1, timeout=timeout)
    assert ok, (
        f"Service {service!r} did not reach expected state {expected_state!r} within {timeout}s. "
        f"Last observed state: {last_state!r}"
    )
