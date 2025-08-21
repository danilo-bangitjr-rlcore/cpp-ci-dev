from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_client(monkeypatch: pytest.MonkeyPatch, dist_with_fake_executable: Path):
    """Spin up the FastAPI app with a temporary base path and yield a TestClient.

    This fixture ensures the module-level app is (re)imported with the desired
    --base-path CLI argument and that any started agents are stopped after the test.
    """
    # Ensure the fake agent stays alive when started
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    # Prepare argv so coredinator.app.parse_base_path() can succeed on import
    prev_argv = sys.argv[:]
    sys.argv = ["pytest", "--base-path", str(dist_with_fake_executable)]

    # Force a clean import of the app module
    module_name = "coredinator.app"
    if module_name in sys.modules:
        del sys.modules[module_name]
    app_module = importlib.import_module(module_name)

    client = TestClient(app_module.app)

    try:
        yield client
    finally:
        # Stop any agents that may still be running
        try:
            manager = app_module.app.state.agent_manager
            for agent_id in list(manager.list_agents()):
                manager.stop_agent(agent_id)
                # Give processes a moment to exit
                time.sleep(0.05)
        except Exception:
            pass
        # Restore argv
        sys.argv = prev_argv



@pytest.mark.timeout(10)
def test_root_redirect_and_version_header(app_client: TestClient):
    # Root should redirect to /docs
    r = app_client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 303, 307, 308)
    assert r.headers.get("location") == "/docs"

    # Version header should be present on all responses
    assert "X-CoreRL-Version" in r.headers
    assert r.headers["X-CoreRL-Version"] != ""


@pytest.mark.timeout(10)
def test_agent_start_status_stop_e2e(app_client: TestClient, config_file: Path):
    # Start the agent via HTTP
    start = app_client.post("/api/agents/start", json={"config_path": str(config_file)})
    assert start.status_code == 200
    agent_id = start.json()
    assert isinstance(agent_id, str)
    assert agent_id == config_file.stem

    # Give a brief moment for processes to boot
    time.sleep(0.2)

    # Check status is running
    status = app_client.get(f"/api/agents/{agent_id}/status")
    assert status.status_code == 200
    body = status.json()
    assert body["id"] == agent_id
    assert body["state"] == "running"
    assert body["config_path"].endswith(config_file.name)

    # List agents includes our agent
    listing = app_client.get("/api/agents/")
    assert listing.status_code == 200
    assert agent_id in listing.json()

    # Stop the agent
    stop = app_client.post(f"/api/agents/{agent_id}/stop")
    assert stop.status_code == 200

    # Status should now be stopped
    time.sleep(0.2)
    status2 = app_client.get(f"/api/agents/{agent_id}/status")
    assert status2.status_code == 200
    assert status2.json()["state"] == "stopped"
