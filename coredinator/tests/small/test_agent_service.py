from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from coredinator.agent.agent_manager import AgentID, AgentManager
from coredinator.service.protocols import ServiceID
from coredinator.service.service import Service, ServiceConfig
from coredinator.service.service_manager import ServiceManager
from coredinator.services.coreio import CoreIOService
from coredinator.services.corerl import CoreRLService
from coredinator.utils.test_polling import wait_for_event


def test_initial_status_stopped(
    dist_with_fake_executable: Path,
):
    """
    Test that the initial status of an AgentProcess is stopped.
    """
    manager = AgentManager(base_path=dist_with_fake_executable, service_manager=ServiceManager())

    agent_id = AgentID("agent")
    s = manager.get_agent_status(agent_id)

    assert s.id == agent_id
    assert s.state == "stopped"
    assert s.service_statuses == {}


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

    manager = AgentManager(base_path=dist_with_fake_executable, service_manager=ServiceManager())
    agent_id = manager.start_agent(config_file)

    # Wait for agent to start
    assert wait_for_event(lambda: manager.get_agent_status(agent_id).state == "running", interval=0.05, timeout=1.0)

    # Check service statuses
    status = manager.get_agent_status(agent_id)
    assert "corerl" in status.service_statuses
    assert "coreio" in status.service_statuses
    assert status.service_statuses["corerl"].state in ["running", "starting"]
    assert status.service_statuses["coreio"].state in ["running", "starting"]

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

    manager = AgentManager(base_path=dist_with_fake_executable, service_manager=ServiceManager())
    agent_id = manager.start_agent(config_file)
    assert wait_for_event(lambda: manager.get_agent_status(agent_id).state == "running", interval=0.05, timeout=1.0)

    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"

    # Stopping again should be idempotent
    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"


@pytest.mark.timeout(10)
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

    manager = AgentManager(base_path=dist_with_fake_executable, service_manager=ServiceManager())
    agent_id = manager.start_agent(config_file)

    # Wait for process to exit with failure
    assert wait_for_event(lambda: manager.get_agent_status(agent_id).state == "failed", interval=0.05, timeout=8.0)


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
    manager = AgentManager(base_path=dist_with_fake_executable, service_manager=ServiceManager())
    agent_id = manager.start_agent(config_file)

    assert wait_for_event(lambda: manager.get_agent_status(agent_id).state == "running", interval=0.05, timeout=1.0)

    # Starting again should be a no-op while running
    manager.start_agent(config_file)

    # Should still be running
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
    manager = AgentManager(base_path=dist_with_fake_executable, service_manager=ServiceManager())
    agent_id = manager.start_agent(config_file)

    assert wait_for_event(lambda: manager.get_agent_status(agent_id).state == "running", interval=0.05, timeout=1.0)

    # Kill one of the services
    # It's a bit ugly to reach into the private attributes, but it's the most
    # direct way to simulate a service crash for this test.
    agent = manager._agents[agent_id]
    coreio_service = manager._service_manager.get_service(agent._coreio_service_id)
    assert coreio_service is not None
    assert isinstance(coreio_service, CoreIOService)
    assert coreio_service._process is not None
    coreio_service._process.kill()

    # Wait for agent to detect the failure
    assert wait_for_event(lambda: manager.get_agent_status(agent_id).state == "failed", interval=0.05, timeout=1.0)
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
    config = ServiceConfig(heartbeat_interval=timedelta(milliseconds=50), degraded_wait=timedelta(milliseconds=200))
    service = service_cls(ServiceID("svc"), config_file, dist_with_fake_executable, config)
    service.start()

    # Wait for degraded recovery logic to trigger
    assert wait_for_event(lambda: service.status().state == "failed", interval=0.05, timeout=1.0)

    # Now simulate healthy agent
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    assert wait_for_event(lambda: service.status().state == "running", interval=0.05, timeout=2.0)
    service.stop()


@pytest.mark.timeout(10)
def test_healthcheck_fails_when_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
    free_localhost_port: int,
):
    """Test that unhealthy healthcheck makes service state failed."""
    # Configure agent to run but be unhealthy
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    monkeypatch.setenv("FAKE_AGENT_HEALTHCHECK", "unhealthy")

    # Enable healthcheck with a dynamically allocated port
    config = ServiceConfig(
        healthcheck_enabled=True,
        port=free_localhost_port,
        healthcheck_timeout=timedelta(seconds=1),
    )
    service = CoreIOService(ServiceID("test-svc"), config_file, dist_with_fake_executable, config)

    # Set the port for the fake agent
    monkeypatch.setenv("FAKE_AGENT_PORT", str(free_localhost_port))

    service.start()

    # Wait for service to start and healthcheck to be performed
    assert wait_for_event(lambda: service.status().state == "failed", interval=0.1, timeout=3.0)

    service.stop()


@pytest.mark.timeout(10)
def test_healthcheck_succeeds_when_healthy(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
    free_localhost_port: int,
):
    """Test that healthy healthcheck makes service state running."""
    # Configure agent to run and be healthy
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    monkeypatch.setenv("FAKE_AGENT_HEALTHCHECK", "healthy")

    # Enable healthcheck with a dynamically allocated port
    config = ServiceConfig(
        healthcheck_enabled=True,
        port=free_localhost_port,
        healthcheck_timeout=timedelta(seconds=1),
    )
    service = CoreIOService(ServiceID("test-svc"), config_file, dist_with_fake_executable, config)

    # Set the port for the fake agent
    monkeypatch.setenv("FAKE_AGENT_PORT", str(free_localhost_port))

    service.start()

    # Wait for service to start and healthcheck to be performed
    assert wait_for_event(lambda: service.status().state == "running", interval=0.1, timeout=3.0)

    service.stop()


@pytest.mark.timeout(10)
def test_healthcheck_disabled_ignores_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
    free_localhost_port: int,
):
    """Test that disabled healthcheck ignores endpoint health."""
    # Configure agent to run but be unhealthy
    monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
    monkeypatch.setenv("FAKE_AGENT_HEALTHCHECK", "unhealthy")

    # Disable healthcheck
    config = ServiceConfig(
        healthcheck_enabled=False,
        port=free_localhost_port,
    )
    service = CoreIOService(ServiceID("test-svc"), config_file, dist_with_fake_executable, config)

    # Set the port for the fake agent
    monkeypatch.setenv("FAKE_AGENT_PORT", str(free_localhost_port))

    service.start()

    # Wait for service to start
    assert wait_for_event(lambda: service.status().state == "running", interval=0.1, timeout=3.0)

    service.stop()
