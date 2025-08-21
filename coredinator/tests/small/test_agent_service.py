from __future__ import annotations

import shutil
import stat
import time
from datetime import timedelta
from pathlib import Path

import pytest

from coredinator.agent.agent_manager import AgentID, AgentManager
from coredinator.service.protocols import ServiceID
from coredinator.service.service import Service, ServiceConfig
from coredinator.services.coreio import CoreIOService
from coredinator.services.corerl import CoreRLService


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "example_config.yaml"
    cfg.write_text("dummy: true\n")
    return cfg


@pytest.fixture()
def dist_with_fake_executable(tmp_path: Path) -> Path:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    src = Path(__file__).parent.parent / "fixtures" / "fake_agent.py"
    dst_coreio = dist_dir / "coreio-1.0.0"
    dst_corerl = dist_dir / "corerl-1.0.0"
    for dst in [dst_coreio, dst_corerl]:
        shutil.copy(src, dst)
        # Make executable
        mode = dst.stat().st_mode
        dst.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dist_dir


def test_initial_status_stopped(
    dist_with_fake_executable: Path,
):
    """
    Test that the initial status of an AgentProcess is stopped.
    """
    manager = AgentManager(base_path=dist_with_fake_executable)

    agent_id = AgentID('agent')
    s = manager.get_agent_status(agent_id)

    assert s.id == agent_id
    assert s.state == "stopped"


@pytest.mark.timeout(5)
def test_start_and_running_status(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that starting an AgentProcess transitions it to running status.
    """
    # Default behavior is long-running process. Ensure environment is set.
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    manager = AgentManager(base_path=dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)

    # Give a brief moment for process to boot
    time.sleep(0.2)

    assert manager.get_agent_status(agent_id).state == "running"

    # Clean up
    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"


@pytest.mark.timeout(5)
def test_stop_transitions_to_stopped(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that stopping an AgentProcess transitions it to stopped status.
    """
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

    manager = AgentManager(base_path=dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)
    time.sleep(0.2)

    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"

    time.sleep(0.2)
    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"


@pytest.mark.timeout(5)
def test_failed_status_when_process_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that an AgentProcess status is failed when the process exits with a non-zero code.
    """
    # Configure the fake agent to exit with failure immediately.
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "exit-1")

    manager = AgentManager(base_path=dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)

    # Wait briefly to allow process to exit
    time.sleep(0.2)

    assert manager.get_agent_status(agent_id).state == "failed"


@pytest.mark.timeout(5)
def test_start_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that starting an AgentProcess is idempotent when already running.
    """
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    manager = AgentManager(base_path=dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)

    time.sleep(0.2)
    assert manager.get_agent_status(agent_id).state == "running"

    # Starting again should be a no-op while running
    manager.start_agent(config_file)

    time.sleep(0.2)
    assert manager.get_agent_status(agent_id).state == "running"

    manager.stop_agent(agent_id)


@pytest.mark.timeout(5)
def test_agent_fails_when_child_service_fails(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that an AgentProcess status is failed if one of its child services fails.
    """
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    manager = AgentManager(base_path=dist_with_fake_executable)
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


@pytest.mark.timeout(5)
@pytest.mark.parametrize("service_cls", [CoreIOService, CoreRLService])
def test_degraded_state_triggers_restart(
    service_cls: type[Service],
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    # Start unhealthy, then become healthy
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "exit-1")
    config = ServiceConfig(heartbeat_interval=timedelta(milliseconds=100), degraded_wait=timedelta(milliseconds=300))
    service = service_cls(ServiceID("svc"), config_file, dist_with_fake_executable, config)
    service.start()

    # Wait for degraded recovery logic to trigger
    time.sleep(0.5)
    assert service.status().state == "failed"

    # Now simulate healthy agent
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    time.sleep(0.5)

    status = service.status()
    assert status.state == "running"
    service.stop()
