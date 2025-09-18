from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from coredinator.app import create_app
from coredinator.utils.test_polling import wait_for_event


@pytest.fixture()
def app_client(monkeypatch: pytest.MonkeyPatch, dist_with_fake_executable: Path):
    """Spin up the FastAPI app with a temporary base path and yield a TestClient.

    This fixture creates a FastAPI app instance with the desired base_path
    and ensures any started agents are stopped after the test.
    """
    # Ensure the fake agent stays alive when started
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    # Prepare argv so coredinator.app.parse_base_path() can succeed on import
    prev_argv = sys.argv[:]
    sys.argv = ["pytest", "--base-path", str(dist_with_fake_executable)]

    # Create app with the test base path
    app = create_app(base_path=dist_with_fake_executable)
    client = TestClient(app)

    try:
        yield client
    finally:
        # Stop any agents that may still be running
        try:
            manager = app.state.agent_manager
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

    # Wait for agent to start
    def _agent_running():
        status = app_client.get(f"/api/agents/{agent_id}/status")
        return status.status_code == 200 and status.json().get("state") == "running"

    assert wait_for_event(_agent_running, interval=0.05, timeout=2.0)

    # Verify status details
    status = app_client.get(f"/api/agents/{agent_id}/status")
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

    # Wait for agent to stop
    def _agent_stopped():
        status = app_client.get(f"/api/agents/{agent_id}/status")
        return status.status_code == 200 and status.json().get("state") == "stopped"

    assert wait_for_event(_agent_stopped, interval=0.05, timeout=1.0)


@pytest.mark.timeout(15)
def test_two_agents_independent_lifecycle(app_client: TestClient, tmp_path: Path):
    # Create two distinct config files
    cfg1 = tmp_path / "agent_one.yaml"
    cfg2 = tmp_path / "agent_two.yaml"
    cfg1.write_text("dummy: 1\n")
    cfg2.write_text("dummy: 2\n")

    # Start both agents
    r1 = app_client.post("/api/agents/start", json={"config_path": str(cfg1)})
    r2 = app_client.post("/api/agents/start", json={"config_path": str(cfg2)})
    assert r1.status_code == 200 and r2.status_code == 200
    id1 = r1.json()
    id2 = r2.json()
    assert id1 == cfg1.stem and id2 == cfg2.stem and id1 != id2

    # Wait for both agents to be running
    def _both_running():
        s1 = app_client.get(f"/api/agents/{id1}/status")
        s2 = app_client.get(f"/api/agents/{id2}/status")
        return (s1.status_code == 200 and s1.json().get("state") == "running" and
                s2.status_code == 200 and s2.json().get("state") == "running")

    assert wait_for_event(_both_running, interval=0.05, timeout=2.0)

    # Listing contains both
    listing = app_client.get("/api/agents/")
    assert listing.status_code == 200
    agents = listing.json()
    assert id1 in agents and id2 in agents

    # Stop first agent; second remains running
    stop1 = app_client.post(f"/api/agents/{id1}/stop")
    assert stop1.status_code == 200

    # Wait for first agent to stop while second remains running
    def _first_stopped_second_running():
        s1 = app_client.get(f"/api/agents/{id1}/status")
        s2 = app_client.get(f"/api/agents/{id2}/status")
        return (s1.status_code == 200 and s1.json().get("state") == "stopped" and
                s2.status_code == 200 and s2.json().get("state") == "running")

    assert wait_for_event(_first_stopped_second_running, interval=0.05, timeout=1.0)

    # Stop second agent
    stop2 = app_client.post(f"/api/agents/{id2}/stop")
    assert stop2.status_code == 200

    # Wait for second agent to stop
    def _second_stopped():
        s2 = app_client.get(f"/api/agents/{id2}/status")
        return s2.status_code == 200 and s2.json().get("state") == "stopped"

    assert wait_for_event(_second_stopped, interval=0.05, timeout=1.0)


# ----------
# Additional E2E coverage
# ----------


@pytest.mark.timeout(5)
def test_agents_list_empty_initially(app_client: TestClient):
    r = app_client.get("/api/agents/")
    assert r.status_code == 200
    assert r.json() == []
    assert "X-CoreRL-Version" in r.headers


@pytest.mark.timeout(5)
def test_start_with_missing_config_returns_400(app_client: TestClient):
    bad_path = "/this/path/does/not/exist.yaml"
    r = app_client.post("/api/agents/start", json={"config_path": bad_path})
    assert r.status_code == 400


@pytest.mark.timeout(5)
def test_start_with_missing_executables_returns_400(
    app_client: TestClient,
    tmp_path: Path,
    dist_with_fake_executable: Path,
):
    # The base_path is the same as dist_with_fake_executable passed to --base-path
    base_path = dist_with_fake_executable
    for name in ("coreio-1.0.0", "corerl-1.0.0"):
        p = base_path / name
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    cfg = tmp_path / "agent_missing_exec.yaml"
    cfg.write_text("a: 1\n")
    r = app_client.post("/api/agents/start", json={"config_path": str(cfg)})
    assert r.status_code == 400


@pytest.mark.timeout(10)
def test_start_idempotent_via_http(app_client: TestClient, config_file: Path):
    r1 = app_client.post("/api/agents/start", json={"config_path": str(config_file)})
    r2 = app_client.post("/api/agents/start", json={"config_path": str(config_file)})
    assert r1.status_code == 200 and r2.status_code == 200
    agent_id_1 = r1.json()
    agent_id_2 = r2.json()
    assert agent_id_1 == agent_id_2 == config_file.stem

    # Wait for agent to be running
    def _agent_running():
        s = app_client.get(f"/api/agents/{agent_id_1}/status")
        return s.status_code == 200 and s.json().get("state") == "running"

    assert wait_for_event(_agent_running, interval=0.05, timeout=2.0)


@pytest.mark.timeout(10)
def test_stop_idempotent_and_unknown_ok(app_client: TestClient, config_file: Path):
    start = app_client.post("/api/agents/start", json={"config_path": str(config_file)})
    assert start.status_code == 200
    agent_id = start.json()

    stop1 = app_client.post(f"/api/agents/{agent_id}/stop")
    stop2 = app_client.post(f"/api/agents/{agent_id}/stop")
    assert stop1.status_code == 200 and stop2.status_code == 200

    # Unknown id stop also returns 200
    stop_unknown = app_client.post("/api/agents/unknown-id/stop")
    assert stop_unknown.status_code == 200

    # Wait for agent to be stopped
    def _agent_stopped():
        status = app_client.get(f"/api/agents/{agent_id}/status")
        return status.status_code == 200 and status.json().get("state") == "stopped"

    assert wait_for_event(_agent_stopped, interval=0.05, timeout=1.0)


@pytest.mark.timeout(5)
def test_status_unknown_agent_returns_stopped(app_client: TestClient):
    unknown = "does-not-exist"
    r = app_client.get(f"/api/agents/{unknown}/status")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == unknown
    assert body["state"] == "stopped"
    assert body["config_path"] is None


@pytest.mark.timeout(10)
def test_failed_agent_status_when_child_exits_nonzero(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Override behavior to exit with failure immediately for this test
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "exit-1")
    cfg = tmp_path / "bad_agent.yaml"
    cfg.write_text("x: y\n")

    start = app_client.post("/api/agents/start", json={"config_path": str(cfg)})
    assert start.status_code == 200
    agent_id = start.json()

    # Wait for agent to fail
    def _agent_failed():
        status = app_client.get(f"/api/agents/{agent_id}/status")
        return status.status_code == 200 and status.json().get("state") == "failed"

    assert wait_for_event(_agent_failed, interval=0.1, timeout=4.0)


@pytest.mark.timeout(5)
def test_cors_preflight_allows_origin(app_client: TestClient):
    headers = {
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "GET",
    }
    r = app_client.options("/api/agents/", headers=headers)
    assert r.status_code in (200, 204)
    # Depending on integration, FastAPI's CORS middleware sets these headers
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example.com")
    assert "GET" in r.headers.get("access-control-allow-methods", "")


@pytest.mark.timeout(10)
def test_list_persists_ids_after_stop(app_client: TestClient, tmp_path: Path):
    cfg1 = tmp_path / "first.yaml"
    cfg2 = tmp_path / "second.yaml"
    cfg1.write_text("a: 1\n")
    cfg2.write_text("b: 2\n")
    id1 = app_client.post("/api/agents/start", json={"config_path": str(cfg1)}).json()
    id2 = app_client.post("/api/agents/start", json={"config_path": str(cfg2)}).json()

    # Wait for both agents to start
    def _both_running():
        s1 = app_client.get(f"/api/agents/{id1}/status")
        s2 = app_client.get(f"/api/agents/{id2}/status")
        return (s1.status_code == 200 and s1.json().get("state") == "running" and
                s2.status_code == 200 and s2.json().get("state") == "running")

    assert wait_for_event(_both_running, interval=0.05, timeout=2.0)

    app_client.post(f"/api/agents/{id1}/stop")
    app_client.post(f"/api/agents/{id2}/stop")

    # Wait for both agents to stop
    def _both_stopped():
        s1 = app_client.get(f"/api/agents/{id1}/status")
        s2 = app_client.get(f"/api/agents/{id2}/status")
        return (s1.status_code == 200 and s1.json().get("state") == "stopped" and
                s2.status_code == 200 and s2.json().get("state") == "stopped")

    assert wait_for_event(_both_stopped, interval=0.05, timeout=1.0)

    listing = app_client.get("/api/agents/").json()
    assert id1 in listing and id2 in listing


@pytest.mark.timeout(10)
def test_same_stem_config_results_in_same_agent_id(app_client: TestClient, tmp_path: Path):
    # Two different directories, same filename
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    f1 = d1 / "agent.yaml"
    f2 = d2 / "agent.yaml"
    f1.write_text("x: 1\n")
    f2.write_text("x: 2\n")

    r1 = app_client.post("/api/agents/start", json={"config_path": str(f1)})
    r2 = app_client.post("/api/agents/start", json={"config_path": str(f2)})
    assert r1.status_code == 200 and r2.status_code == 200
    id1 = r1.json()
    id2 = r2.json()
    assert id1 == id2 == "agent"

    # List should only contain one entry for that id
    agents = app_client.get("/api/agents/").json()
    assert agents.count("agent") == 1
