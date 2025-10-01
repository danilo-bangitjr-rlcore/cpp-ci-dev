"""HTTP API client utilities for coredinator testing."""

from typing import Any

import requests

from tests.utils.state_verification import assert_agent_http_state, wait_for_agent_http_state
from tests.utils.types import AgentState


class CoredinatorAPIClient:
    """HTTP API client for coredinator testing with standardized request patterns."""

    def __init__(self, base_url: str, request_timeout: float = 30.0):
        self.base_url = base_url
        self.request_timeout = request_timeout

    def start_agent(self, config_path: str, coreio_id: str | None = None) -> str:
        """Start an agent and return its ID."""
        payload = {"config_path": config_path}
        if coreio_id:
            payload["coreio_id"] = coreio_id

        response = requests.post(f"{self.base_url}/api/agents/start", json=payload, timeout=self.request_timeout)
        assert response.status_code == 200, (
            f"Failed to start agent: status={response.status_code}, body={response.text}"
        )
        return response.json()

    def stop_agent(self, agent_id: str) -> None:
        """Stop an agent."""
        response = requests.post(f"{self.base_url}/api/agents/{agent_id}/stop", timeout=self.request_timeout)
        assert response.status_code == 200, (
            f"Failed to stop agent {agent_id}: status={response.status_code}, body={response.text}"
        )

    def get_agent_status(self, agent_id: str) -> dict[str, Any]:
        """Get agent status information."""
        response = requests.get(f"{self.base_url}/api/agents/{agent_id}/status", timeout=self.request_timeout)
        assert response.status_code == 200, (
            f"Failed to get agent status for {agent_id}: status={response.status_code}, body={response.text}"
        )
        return response.json()

    def start_coreio_service(self, config_path: str, coreio_id: str | None = None) -> str:
        """Start a CoreIO service and return its service ID."""
        payload = {"config_path": config_path}
        if coreio_id:
            payload["coreio_id"] = coreio_id

        response = requests.post(f"{self.base_url}/api/io/start", json=payload, timeout=self.request_timeout)
        assert response.status_code == 200, (
            f"Failed to start CoreIO: status={response.status_code}, body={response.text}"
        )
        return response.json()["service_id"]

    def stop_coreio_service(self, service_id: str) -> None:
        """Stop a CoreIO service."""
        response = requests.post(f"{self.base_url}/api/io/{service_id}/stop", timeout=self.request_timeout)
        assert response.status_code == 200, (
            f"Failed to stop CoreIO {service_id}: status={response.status_code}, body={response.text}"
        )

    def get_coreio_status(self, service_id: str) -> dict[str, Any]:
        """Get CoreIO service status information."""
        response = requests.get(f"{self.base_url}/api/io/{service_id}/status", timeout=self.request_timeout)
        assert response.status_code == 200, (
            f"Failed to get CoreIO status for {service_id}: status={response.status_code}, body={response.text}"
        )
        return response.json()

    def wait_for_agent_state(self, agent_id: str, expected_state: AgentState, timeout: float = 2.0) -> bool:
        """Wait for agent to reach expected state."""
        return wait_for_agent_http_state(self.base_url, agent_id, expected_state, timeout)

    def assert_agent_state(self, agent_id: str, expected_state: AgentState, timeout: float = 2.0) -> None:
        """Assert that agent reaches expected state within timeout."""
        assert_agent_http_state(self.base_url, agent_id, expected_state, timeout)
