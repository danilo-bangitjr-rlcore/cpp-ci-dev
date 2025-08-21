from __future__ import annotations

import stat
import time
from pathlib import Path

import pytest

from coredinator.agent.agent_manager import AgentID, AgentManager


@pytest.fixture(scope="session")
def fake_agent_path() -> Path:
    here = Path(__file__).parent
    agent = here / "fixtures" / "fake_agent.py"
    # Ensure executable bit for POSIX
    mode = agent.stat().st_mode
    agent.chmod(mode | stat.S_IXUSR)
    return agent


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "example_config.yaml"
    cfg.write_text("dummy: true\n")
    return cfg


@pytest.fixture(autouse=True)
def patch_service_executables(monkeypatch: pytest.MonkeyPatch, fake_agent_path: Path):
    monkeypatch.setattr("coredinator.services.corerl.CoreRLService.EXECUTABLE_PATH", fake_agent_path)
    monkeypatch.setattr("coredinator.services.coreio.CoreIOService.EXECUTABLE_PATH", fake_agent_path)


def test_initial_status_stopped(config_file: Path):
    """
    Test that the initial status of an AgentProcess is stopped.
    """
    manager = AgentManager()

    agent_id = AgentID('agent')
    s = manager.get_agent_status(agent_id)

    assert s.id == agent_id
    assert s.state == "stopped"


@pytest.mark.timeout(5)
def test_start_and_running_status(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    """
    Test that starting an AgentProcess transitions it to running status.
    """
    # Default behavior is long-running process. Ensure environment is set.
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    manager = AgentManager()
    agent_id = manager.start_agent(config_file)

    # Give a brief moment for process to boot
    time.sleep(0.2)

    assert manager.get_agent_status(agent_id).state == "running"

    # Clean up
    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"


@pytest.mark.timeout(5)
def test_stop_transitions_to_stopped(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    """
    Test that stopping an AgentProcess transitions it to stopped status.
    """
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    manager = AgentManager()
    agent_id = manager.start_agent(config_file)
    time.sleep(0.2)

    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"

    time.sleep(0.2)
    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"


@pytest.mark.timeout(5)
def test_failed_status_when_process_exits_nonzero(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    """
    Test that an AgentProcess status is failed when the process exits with a non-zero code.
    """
    # Configure the fake agent to exit with failure immediately.
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "exit-1")

    manager = AgentManager()
    agent_id = manager.start_agent(config_file)

    # Wait briefly to allow process to exit
    time.sleep(0.2)

    assert manager.get_agent_status(agent_id).state == "failed"


@pytest.mark.timeout(5)
def test_start_is_idempotent(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    """
    Test that starting an AgentProcess is idempotent when already running.
    """
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    manager = AgentManager()
    agent_id = manager.start_agent(config_file)

    time.sleep(0.2)
    assert manager.get_agent_status(agent_id).state == "running"

    # Starting again should be a no-op while running
    manager.start_agent(config_file)

    time.sleep(0.2)
    assert manager.get_agent_status(agent_id).state == "running"

    manager.stop_agent(agent_id)


@pytest.mark.timeout(5)
def test_agent_fails_when_child_service_fails(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    manager = AgentManager()
    agent_id = manager.start_agent(config_file)

    time.sleep(0.2)
    assert manager.get_agent_status(agent_id).state == "running"

    # Kill one of the services
    # It's a bit ugly to reach into the private attributes, but it's the most
    # direct way to simulate a service crash for this test.
    coreio_process = manager._agents[agent_id]._coreio_service._process
    assert coreio_process is not None
    coreio_process.kill()

    time.sleep(0.2)  # Allow time for status to update

    assert manager.get_agent_status(agent_id).state == "failed"
    manager.stop_agent(agent_id)
