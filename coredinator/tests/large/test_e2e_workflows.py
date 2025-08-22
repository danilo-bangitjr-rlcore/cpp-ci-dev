"""Large integration tests for coredinator service.

These tests start the coredinator service as a subprocess and make real HTTP
requests to test end-to-end workflows.
"""
import pytest
import requests


@pytest.mark.timeout(15)
def test_agent_status_unknown_returns_stopped(coredinator_service: str):
    """Test that querying status for unknown agent returns 'stopped' state."""
    base_url = coredinator_service
    unknown_agent_id = "nonexistent-agent"

    # Query status for an agent that doesn't exist
    response = requests.get(f"{base_url}/api/agents/{unknown_agent_id}/status")

    # Should return 200 with stopped state
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == unknown_agent_id
    assert data["state"] == "stopped"
    assert data["config_path"] is None
