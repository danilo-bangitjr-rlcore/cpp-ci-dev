from __future__ import annotations

import stat
import time
from pathlib import Path

import pytest

from coredinator.agent.agent_process import AgentID, AgentProcess


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
def patch_agent_executable(monkeypatch: pytest.MonkeyPatch, fake_agent_path: Path):
    from coredinator.agent import agent_process as ap

    monkeypatch.setattr(ap, "AGENT_EXECUTABLE", fake_agent_path)


def test_initial_status_stopped(config_file: Path):
    """
    Test that the initial status of an AgentProcess is stopped.
    """
    proc = AgentProcess(id=AgentID("a1"), config_path=config_file)
    s = proc.status()
    assert s.id == "a1"
    assert s.state == "stopped"
    assert s.config_path == config_file


@pytest.mark.timeout(5)
def test_start_and_running_status(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    """
    Test that starting an AgentProcess transitions it to running status.
    """
    # Default behavior is long-running process. Ensure environment is set.
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    proc = AgentProcess(id=AgentID("a2"), config_path=config_file)
    proc.start()

    # Give a brief moment for process to boot
    time.sleep(0.2)

    assert proc.status().state == "running"

    # Clean up
    proc.stop()
    assert proc.status().state == "stopped"


@pytest.mark.timeout(5)
def test_stop_transitions_to_stopped(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    """
    Test that stopping an AgentProcess transitions it to stopped status.
    """
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    proc = AgentProcess(id=AgentID("a3"), config_path=config_file)
    proc.start()
    time.sleep(0.2)

    proc.stop()
    assert proc.status().state == "stopped"

    time.sleep(0.2)
    proc.stop()
    assert proc.status().state == "stopped"


@pytest.mark.timeout(5)
def test_failed_status_when_process_exits_nonzero(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    """
    Test that an AgentProcess status is failed when the process exits with a non-zero code.
    """
    # Configure the fake agent to exit with failure immediately.
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "exit-1")

    proc = AgentProcess(id=AgentID("a4"), config_path=config_file)
    proc.start()

    # Wait briefly to allow process to exit
    time.sleep(0.2)

    assert proc.status().state == "failed"


@pytest.mark.timeout(5)
def test_start_is_idempotent(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    """
    Test that starting an AgentProcess is idempotent when already running.
    """
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    proc = AgentProcess(id=AgentID("a5"), config_path=config_file)

    proc.start()

    time.sleep(0.2)
    assert proc.status().state == "running"

    # Starting again should be a no-op while running
    proc.start()

    time.sleep(0.2)
    assert proc.status().state == "running"

    proc.stop()
