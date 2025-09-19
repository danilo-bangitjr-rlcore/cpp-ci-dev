from pathlib import Path

import pytest

from coredinator.agent.agent import Agent, AgentID
from coredinator.service.protocols import ServiceBundleID, ServiceID, ServiceState
from coredinator.service.service_manager import ServiceManager


class TestAgentSharedServices:
    """Test shared service behavior in Agent class."""

    def test_agent_registers_with_service_manager(
        self, config_file: Path, dist_with_fake_executable: Path,
    ):
        """
        Test that Agent registers itself with ServiceManager.
        """
        service_manager = ServiceManager(dist_with_fake_executable)
        agent = Agent(  # noqa: F841  # Agent must stay in scope for ServiceManager registration
            id=AgentID("test-agent"),
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
        )

        # Check that agent is registered and its services are tracked
        bundle_services = service_manager.get_bundle_services(ServiceBundleID("test-agent"))
        assert ServiceID("test-agent-corerl") in bundle_services
        assert ServiceID("test-agent-coreio") in bundle_services

        # Check that services have the agent as owner
        assert ServiceBundleID("test-agent") in service_manager.get_service_owners(ServiceID("test-agent-corerl"))
        assert ServiceBundleID("test-agent") in service_manager.get_service_owners(ServiceID("test-agent-coreio"))

    @pytest.mark.timeout(10)
    def test_agent_stop_uses_service_manager(
        self, monkeypatch: pytest.MonkeyPatch, config_file: Path, dist_with_fake_executable: Path,
    ):
        """
        Test that Agent.stop() delegates to ServiceManager properly.
        """
        monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

        service_manager = ServiceManager(dist_with_fake_executable)
        agent = Agent(
            id=AgentID("test-agent"),
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
        )

        # Start the agent to create real services
        agent.start()

        # Get the services before stopping
        corerl_service = service_manager.get_service(ServiceID("test-agent-corerl"))
        coreio_service = service_manager.get_service(ServiceID("test-agent-coreio"))
        assert corerl_service is not None
        assert coreio_service is not None

        # Stop the agent - this should unregister and stop services
        agent.stop(grace_seconds=1.0)

        # Verify services are no longer owned by the agent
        assert ServiceBundleID("test-agent") not in service_manager.get_service_owners(ServiceID("test-agent-corerl"))
        assert ServiceBundleID("test-agent") not in service_manager.get_service_owners(ServiceID("test-agent-coreio"))


class TestSharedServiceIntegration:
    """Integration tests for shared service management."""

    @pytest.mark.timeout(15)
    def test_shared_service_lifecycle_with_real_agents(
        self, monkeypatch: pytest.MonkeyPatch, config_file: Path, dist_with_fake_executable: Path,
    ) -> None:
        """
        Test complete shared service lifecycle with real Agent instances.
        """
        monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
        service_manager = ServiceManager(dist_with_fake_executable)

        # Create first agent with standard service IDs
        agent1 = Agent(
            id=AgentID("shared-demo"),
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
        )

        # Create second agent that shares CoreIO service with agent1
        shared_coreio_id = ServiceID("shared-demo-coreio")
        agent2 = Agent(
            id=AgentID("other-agent"),
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
            coreio_service_id=shared_coreio_id,
        )

        # Verify shared service setup
        assert service_manager.is_service_shared(shared_coreio_id)
        assert not service_manager.is_service_shared(ServiceID("shared-demo-corerl"))
        assert not service_manager.is_service_shared(ServiceID("other-agent-corerl"))

        # Start both agents
        agent1.start()
        agent2.start()

        # Verify both agents report correct status
        status1 = agent1.status()
        status2 = agent2.status()
        assert status1.id == AgentID("shared-demo")
        assert status2.id == AgentID("other-agent")

        # Stop agent1 - shared CoreIO should remain running for agent2
        agent1.stop()

        # Verify CoreIO is no longer shared (only agent2 owns it now)
        assert not service_manager.is_service_shared(shared_coreio_id)
        assert ServiceBundleID("other-agent") in service_manager.get_service_owners(shared_coreio_id)

        # Stop agent2 - now CoreIO should be fully stopped
        agent2.stop()

        # Verify both agents are cleaned up
        assert ServiceBundleID("shared-demo") not in service_manager.get_service_owners(shared_coreio_id)
        assert ServiceBundleID("other-agent") not in service_manager.get_service_owners(shared_coreio_id)

        # Verify CoreIO service is actually stopped
        coreio_service = service_manager.get_service(shared_coreio_id)
        assert coreio_service is not None
        assert coreio_service.status().state == ServiceState.STOPPED

    def test_agent_cleanup_on_destruction(
        self, config_file: Path, dist_with_fake_executable: Path,
    ) -> None:
        """
        Test that agents properly clean up when destroyed.
        """
        service_manager = ServiceManager(dist_with_fake_executable)

        # Create agent in a scope
        agent_id = AgentID("cleanup-test")
        agent = Agent(
            id=agent_id,
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
        )

        # Verify agent is registered
        bundle_id = ServiceBundleID("cleanup-test")
        assert len(service_manager.get_bundle_services(bundle_id)) == 2

        # Delete the agent
        del agent

        # Verify cleanup was attempted
        assert len(service_manager.get_bundle_services(bundle_id)) == 0

    @pytest.mark.timeout(10)
    def test_service_manager_atomic_operations(
        self, monkeypatch: pytest.MonkeyPatch, config_file: Path, dist_with_fake_executable: Path,
    ) -> None:
        """
        Test that ServiceManager operations are properly atomic.
        """
        monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
        service_manager = ServiceManager(dist_with_fake_executable)

        agent = Agent(
            id=AgentID("atomic-test"),
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
        )

        # Start the agent to create real services
        agent.start()

        # Test atomic unregister and stop
        corerl_service = service_manager.get_service(ServiceID("atomic-test-corerl"))
        coreio_service = service_manager.get_service(ServiceID("atomic-test-coreio"))
        assert corerl_service is not None
        assert coreio_service is not None

        stopped_services = service_manager.unregister_bundle(
            ServiceBundleID("atomic-test"), grace_seconds=1.0,
        )

        # Verify return value contains stopped services
        assert ServiceID("atomic-test-corerl") in stopped_services
        assert ServiceID("atomic-test-coreio") in stopped_services
        assert len(stopped_services) == 2

        # Verify services are no longer owned
        assert ServiceBundleID("atomic-test") not in service_manager.get_service_owners(ServiceID("atomic-test-corerl"))
        assert ServiceBundleID("atomic-test") not in service_manager.get_service_owners(ServiceID("atomic-test-coreio"))

    @pytest.mark.timeout(10)
    def test_process_reattachment_with_shared_services(
        self, monkeypatch: pytest.MonkeyPatch, config_file: Path, dist_with_fake_executable: Path,
    ) -> None:
        """
        Test process reattachment functionality with shared services.
        """
        monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
        service_manager = ServiceManager(dist_with_fake_executable)

        agent = Agent(
            id=AgentID("reattach-test"),
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
        )

        # Start agent to create real processes
        agent.start()

        # Get the actual PIDs
        corerl_service = service_manager.get_service(ServiceID("reattach-test-corerl"))
        coreio_service = service_manager.get_service(ServiceID("reattach-test-coreio"))
        assert corerl_service is not None
        assert coreio_service is not None

        # Test reattachment with None PIDs
        corerl_success, coreio_success = agent.reattach_processes(corerl_pid=None, coreio_pid=None)
        assert corerl_success is False
        assert coreio_success is False

        # Clean up
        agent.stop()

    @pytest.mark.timeout(10)
    def test_agent_status_aggregation(
        self, monkeypatch: pytest.MonkeyPatch, config_file: Path, dist_with_fake_executable: Path,
    ) -> None:
        """
        Test agent status aggregation from multiple services.
        """
        monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")
        service_manager = ServiceManager(dist_with_fake_executable)

        agent = Agent(
            id=AgentID("status-test"),
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
        )

        # Test stopped state (before starting)
        status = agent.status()
        assert status.state == ServiceState.STOPPED

        # Start agent and test running state
        agent.start()
        status = agent.status()
        assert status.state == ServiceState.RUNNING

        # Clean up
        agent.stop()
        status = agent.status()
        assert status.state == ServiceState.STOPPED

    def test_error_handling_in_destructor(
        self, config_file: Path, dist_with_fake_executable: Path,
    ) -> None:
        """
        Test that errors in __del__ are properly handled.
        """
        service_manager = ServiceManager(dist_with_fake_executable)

        agent = Agent(
            id=AgentID("error-test"),
            config_path=config_file,
            base_path=dist_with_fake_executable,
            service_manager=service_manager,
        )

        # The destructor should handle errors gracefully
        try:
            del agent
            import gc
            gc.collect()
        except Exception as e:
            pytest.fail(f"Agent destruction should not raise exceptions, but got: {e}")
