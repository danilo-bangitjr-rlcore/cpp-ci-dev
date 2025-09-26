from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from coredinator.agent.agent_manager import AgentID
from coredinator.service import service as service_module
from coredinator.service.protocols import ServiceID
from coredinator.service.service import Service, ServiceConfig
from coredinator.services.coreio import CoreIOService
from coredinator.services.corerl import CoreRLService
from tests.utils.factories import create_agent_manager
from tests.utils.state_verification import (
    assert_agent_manager_state,
    assert_service_state,
)


def _service_for_spawn(monkeypatch: pytest.MonkeyPatch) -> Service:
    service = Service(ServiceID("svc"), Path("exe"), Path("cfg"))
    service._ensure_executable = lambda: Path("exe")
    service._ensure_config = lambda: Path("cfg")
    service._keep_alive = lambda: None
    monkeypatch.setattr(service_module.psutil, "Process", lambda pid: SimpleNamespace(pid=pid))
    return service


def test_service_start_sets_windows_creationflags(monkeypatch: pytest.MonkeyPatch):
    captured_kwargs: list[dict[str, object]] = []

    def fake_popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        captured_kwargs.append(kwargs)
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(service_module, "Popen", fake_popen)
    monkeypatch.setattr(service_module, "IS_WINDOWS", True, raising=False)
    fake_subprocess = SimpleNamespace(DETACHED_PROCESS=0x00000008, CREATE_NEW_PROCESS_GROUP=0x00000200)
    monkeypatch.setattr(service_module, "subprocess", fake_subprocess, raising=False)

    service = _service_for_spawn(monkeypatch)
    service.start()

    assert captured_kwargs
    kwargs = captured_kwargs[0]
    expected_flags = fake_subprocess.DETACHED_PROCESS | fake_subprocess.CREATE_NEW_PROCESS_GROUP
    assert kwargs["creationflags"] == expected_flags
    assert kwargs["start_new_session"] is True


def test_service_start_omits_creationflags_on_non_windows(monkeypatch: pytest.MonkeyPatch):
    captured_kwargs: list[dict[str, object]] = []

    def fake_popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        captured_kwargs.append(kwargs)
        return SimpleNamespace(pid=5678)

    monkeypatch.setattr(service_module, "Popen", fake_popen)
    monkeypatch.setattr(service_module, "IS_WINDOWS", False, raising=False)

    service = _service_for_spawn(monkeypatch)
    service.start()

    assert captured_kwargs
    kwargs = captured_kwargs[0]
    assert "creationflags" not in kwargs
    assert kwargs["start_new_session"] is True


def test_initial_status_stopped(
    dist_with_fake_executable: Path,
):
    """
    Test that the initial status of an AgentProcess is stopped.
    """
    manager = create_agent_manager(dist_with_fake_executable)

    agent_id = AgentID("agent")
    s = manager.get_agent_status(agent_id)

    assert s.id == agent_id
    assert s.state == "stopped"
    assert s.service_statuses == {}


@pytest.mark.timeout(10)
def test_start_and_running_status(
    long_running_agent_env: None,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that starting an AgentProcess transitions it to running status.
    """
    manager = create_agent_manager(dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)

    # Wait for agent to start
    assert_agent_manager_state(manager, agent_id, "running", timeout=2.0)

    # Check service statuses
    status = manager.get_agent_status(agent_id)
    assert "corerl" in status.service_statuses
    assert "coreio" in status.service_statuses
    assert status.service_statuses["corerl"].state in ["running", "starting"]
    assert status.service_statuses["coreio"].state in ["running", "starting"]

    # Clean up
    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"


@pytest.mark.timeout(10)
def test_stop_transitions_to_stopped(
    long_running_agent_env: None,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that stopping an AgentProcess transitions it to stopped status.
    """
    manager = create_agent_manager(dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)
    assert_agent_manager_state(manager, agent_id, "running", timeout=2.0)

    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"

    # Stopping again should be idempotent
    manager.stop_agent(agent_id)
    assert manager.get_agent_status(agent_id).state == "stopped"


@pytest.mark.timeout(15)
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

    manager = create_agent_manager(dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)

    # Wait for process to exit with failure
    assert_agent_manager_state(manager, agent_id, "failed", timeout=12.0)


@pytest.mark.timeout(10)
def test_start_is_idempotent(
    long_running_agent_env: None,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that starting an AgentProcess is idempotent when already running.
    """
    manager = create_agent_manager(dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)

    assert_agent_manager_state(manager, agent_id, "running", timeout=2.0)

    # Starting again should be a no-op while running
    manager.start_agent(config_file)

    # Should still be running
    assert manager.get_agent_status(agent_id).state == "running"

    manager.stop_agent(agent_id)


@pytest.mark.timeout(10)
def test_agent_fails_when_child_service_fails(
    long_running_agent_env: None,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    """
    Test that an AgentProcess status is failed if one of its child services fails.
    """
    manager = create_agent_manager(dist_with_fake_executable)
    agent_id = manager.start_agent(config_file)

    assert_agent_manager_state(manager, agent_id, "running", timeout=2.0)

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
    assert_agent_manager_state(manager, agent_id, "failed", timeout=2.0)
    manager.stop_agent(agent_id)


@pytest.mark.timeout(15)
@pytest.mark.parametrize("service_cls", [CoreIOService, CoreRLService])
def test_degraded_state_triggers_restart(
    service_cls: type[Service],
    long_running_agent_env: None,
    config_file: Path,
    dist_with_fake_executable: Path,
):
    # Test degraded state detection and restart behavior by manually killing process
    config = ServiceConfig(heartbeat_interval=timedelta(milliseconds=50), degraded_wait=timedelta(milliseconds=200))
    service = service_cls(ServiceID("svc"), config_file, dist_with_fake_executable, config)
    service.start()

    # Wait for service to start successfully
    assert_service_state(service, "running", timeout=4.0)

    # Simulate degraded state by killing the process manually
    if service._process is not None:
        service._process.terminate()
        service._process.wait()  # Ensure process is fully terminated

    # Wait for service to detect failure
    assert_service_state(service, "failed", timeout=2.0)

    # Wait for degraded recovery logic to restart the service
    assert_service_state(service, "running", timeout=5.0)
    service.stop()


@pytest.mark.timeout(15)
def test_healthcheck_fails_when_unhealthy(
    long_running_agent_env: None,
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
    free_localhost_port: int,
):
    """Test that unhealthy healthcheck makes service state failed."""
    # Configure agent to be unhealthy
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
    assert_service_state(service, "failed", timeout=5.0)

    service.stop()


@pytest.mark.timeout(15)
def test_healthcheck_succeeds_when_healthy(
    long_running_agent_env: None,
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
    free_localhost_port: int,
):
    """Test that healthy healthcheck makes service state running."""
    # Configure agent to be healthy
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
    assert_service_state(service, "running", timeout=5.0)

    service.stop()


@pytest.mark.timeout(15)
def test_healthcheck_disabled_ignores_endpoint(
    long_running_agent_env: None,
    monkeypatch: pytest.MonkeyPatch,
    config_file: Path,
    dist_with_fake_executable: Path,
    free_localhost_port: int,
):
    """Test that disabled healthcheck ignores endpoint health."""
    # Configure agent to be unhealthy
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
    assert_service_state(service, "running", timeout=5.0)

    service.stop()
