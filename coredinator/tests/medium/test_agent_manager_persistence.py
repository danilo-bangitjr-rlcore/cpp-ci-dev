import sqlite3
from pathlib import Path

import pytest

from coredinator.agent.agent_manager import AgentManager
from coredinator.service.service_manager import ServiceManager
from tests.utils.polling import wait_for_event


class TestAgentManagerPersistence:
    @pytest.mark.timeout(45)
    def test_agent_state_persistence_across_restarts(
        self,
        long_running_agent_env: None,
        tmp_path: Path,
        config_file: Path,
        dist_with_fake_executable: Path,
    ):
        """Test that agent states are properly persisted and restored across AgentManager restarts."""

        # Use dist_with_fake_executable as base_path but our config in tmp_path
        test_config = tmp_path / "test_config.yaml"
        test_config.write_text("dummy: true\n")

        # Phase 1: Start an agent and verify it's running
        manager1 = AgentManager(
            base_path=dist_with_fake_executable,
            service_manager=ServiceManager(base_path=dist_with_fake_executable),
        )
        agent_id = manager1.start_agent(test_config)

        # Wait for agent to start
        assert wait_for_event(
            lambda: manager1.get_agent_status(agent_id).state == "running",
            interval=0.05,
            timeout=6.0,
        )
        assert agent_id in manager1.list_agents()

        # Phase 2: Create new AgentManager (simulating restart)
        # The previous manager's agents should be restored and auto-started
        manager2 = AgentManager(
            base_path=dist_with_fake_executable,
            service_manager=ServiceManager(base_path=dist_with_fake_executable),
        )

        # Verify agent was restored
        assert agent_id in manager2.list_agents()

        # Wait for auto-start to complete
        assert wait_for_event(
            lambda: manager2.get_agent_status(agent_id).state == "running",
            interval=0.05,
            timeout=8.0,
        ), "Agent should auto-start because it was marked as running"

        # Clean up
        manager2.stop_agent(agent_id)

    @pytest.mark.timeout(45)
    def test_multiple_agents_persistence(
        self,
        long_running_agent_env: None,
        tmp_path: Path,
        dist_with_fake_executable: Path,
    ):
        """Test persistence with multiple agents in different states."""

        # Create multiple config files
        config1 = tmp_path / "agent1.yaml"
        config2 = tmp_path / "agent2.yaml"
        config3 = tmp_path / "agent3.yaml"

        for config in [config1, config2, config3]:
            config.write_text("dummy: true\n")

        # Phase 1: Start multiple agents and set different states
        manager1 = AgentManager(
            base_path=dist_with_fake_executable,
            service_manager=ServiceManager(base_path=dist_with_fake_executable),
        )

        agent1_id = manager1.start_agent(config1)  # Will be running
        agent2_id = manager1.start_agent(config2)  # Will be stopped
        agent3_id = manager1.start_agent(config3)  # Will be running

        # Wait for all agents to start
        def _all_running():
            return (
                manager1.get_agent_status(agent1_id).state == "running" and
                manager1.get_agent_status(agent2_id).state == "running" and
                manager1.get_agent_status(agent3_id).state == "running"
            )

        assert wait_for_event(_all_running, interval=0.05, timeout=8.0)

        # Stop agent2 to create mixed states
        manager1.stop_agent(agent2_id)

        # Phase 2: Restart AgentManager and verify correct restoration
        manager2 = AgentManager(
            base_path=dist_with_fake_executable,
            service_manager=ServiceManager(base_path=dist_with_fake_executable),
        )

        # All agents should be restored
        restored_agents = manager2.list_agents()
        assert agent1_id in restored_agents
        assert agent2_id in restored_agents
        assert agent3_id in restored_agents

        # Wait for auto-start to complete and verify states
        def _correct_states():
            return (
                manager2.get_agent_status(agent1_id).state == "running" and
                manager2.get_agent_status(agent2_id).state == "stopped" and  # Should NOT be auto-started
                manager2.get_agent_status(agent3_id).state == "running"
            )

        assert wait_for_event(_correct_states, interval=0.05, timeout=10.0)

        # Clean up
        for agent_id in [agent1_id, agent2_id, agent3_id]:
            manager2.stop_agent(agent_id)


    @pytest.mark.timeout(45)
    def test_empty_database_initialization(self, tmp_path: Path):
        """Test that AgentManager works correctly when starting with an empty database."""
        # Create AgentManager with empty database
        manager = AgentManager(base_path=tmp_path, service_manager=ServiceManager(base_path=tmp_path))

        # Should start with no agents
        assert manager.list_agents() == []

        # Database should exist but be empty
        db_path = tmp_path / "agent_state.db"
        assert db_path.exists()

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_states")
            count = cursor.fetchone()[0]
            assert count == 0

    @pytest.mark.timeout(45)
    def test_database_corruption_recovery(self, tmp_path: Path, config_file: Path):
        """Test that AgentManager handles database file corruption gracefully."""
        db_path = tmp_path / "agent_state.db"

        # Create a corrupted database file
        db_path.write_text("This is not a valid SQLite database")

        manager = AgentManager(base_path=tmp_path, service_manager=ServiceManager(base_path=tmp_path))
        agents = manager.list_agents()
        assert isinstance(agents, list)
