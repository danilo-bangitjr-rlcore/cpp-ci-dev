import sqlite3
from pathlib import Path

import pytest

from coredinator.agent.agent_manager import AgentManager


class TestAgentManagerPersistenceEdgeCases:
    def test_nonexistent_config_path_in_database(self, tmp_path: Path):
        """Test behavior when database contains path to nonexistent config file."""
        db_path = tmp_path / "agent_state.db"

        # Manually create database with nonexistent config path
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE agent_states (
                    agent_id TEXT PRIMARY KEY,
                    operating_mode TEXT NOT NULL CHECK (operating_mode IN ('running', 'stopped')),
                    config_path TEXT NOT NULL,
                    corerl_process_id INTEGER,
                    coreio_process_id INTEGER
                )
            """)
            cursor.execute(
                "INSERT INTO agent_states VALUES (?, ?, ?, ?, ?)",
                ("test_agent", "running", str(tmp_path / "nonexistent.yaml"), None, None),
            )
            conn.commit()

        # AgentManager should fail on startup
        # TODO: consider gracefully marking agent as failed in this case
        try:
            AgentManager(base_path=tmp_path)
        except Exception as e:
            # Should be an informative error about missing config
            assert "config" in str(e).lower() or "file" in str(e).lower()

    def test_database_permissions_error(self, tmp_path: Path):
        """Test behavior when database file cannot be created due to permissions."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        try:
            AgentManager(base_path=readonly_dir)
        except Exception as e:
            # Should be a clear error about permissions or database creation
            assert any(word in str(e).lower() for word in ["permission", "database", "create", "write"])
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)


    def test_idempotent_start_agent_database_consistency(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dist_with_fake_executable: Path,
    ):
        """Test that calling start_agent multiple times doesn't create duplicate database entries."""
        monkeypatch.setenv("FAKE_AGENT_BEHAVIOR", "long")

        config = tmp_path / "test.yaml"
        config.write_text("dummy: true\n")

        manager = AgentManager(base_path=dist_with_fake_executable)

        # Start the same agent multiple times
        agent_id1 = manager.start_agent(config)
        agent_id2 = manager.start_agent(config)  # Should be same agent
        agent_id3 = manager.start_agent(config)  # Should be same agent

        assert agent_id1 == agent_id2 == agent_id3

        # Check database has only one entry
        db_path = dist_with_fake_executable / "agent_state.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_states WHERE agent_id = ?", (agent_id1,))
            count = cursor.fetchone()[0]
            assert count == 1, "Should have exactly one database entry per agent"

        # Clean up
        manager.stop_agent(agent_id1)
