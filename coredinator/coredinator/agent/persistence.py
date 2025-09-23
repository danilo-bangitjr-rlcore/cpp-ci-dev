import sqlite3
from pathlib import Path
from typing import Any

import backoff

from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceIntendedState


class AgentPersistenceLayer:
    """Handles persistence of agent state and configuration."""

    def __init__(self, db_path: Path):
        self._logger = get_logger(__name__)
        self._db_path = db_path
        self._init_database()

    def _init_database(self, retries: int = 1) -> None:
        """Initialize the agent persistence database."""
        self._logger.info(
            "Initializing agent persistence database",
            db_path=str(self._db_path),
            retries_remaining=retries,
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=1,
            on_backoff=self._on_database_init_backoff,
            on_giveup=self._on_database_init_giveup,
        )
        def _do_init_database():
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                self._create_agent_tables(cursor)
                conn.commit()

        _do_init_database()

    def _on_database_init_backoff(self, details: Any):
        """Handle database initialization backoff - remove corrupted database."""
        self._logger.warning(
            "Agent database initialization failed",
            db_path=str(self._db_path),
            retries_remaining=details["tries"] - 1,
        )
        self._logger.info(
            "Removing corrupted agent database and retrying",
            db_path=str(self._db_path),
        )
        self._db_path.unlink(missing_ok=True)

    def _on_database_init_giveup(self, details: Any):
        """Handle final database initialization failure."""
        self._logger.error(
            "Agent database initialization failed permanently",
            db_path=str(self._db_path),
        )

    def _recover_database_and_log(self, agent_id: str | None, operation: str) -> None:
        """Recover database after error and log the attempt."""
        if agent_id:
            self._logger.warning(f"Failed to {operation}", agent_id=agent_id)
        else:
            self._logger.warning(f"Failed to {operation}")
        self._init_database()

    def _create_agent_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create database tables for agent persistence."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT PRIMARY KEY,
                config_path TEXT NOT NULL,
                intended_state TEXT NOT NULL CHECK (
                    intended_state IN ('running', 'stopped')
                ),
                coreio_service_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def persist_agent(
        self,
        agent_id: str,
        config_path: str,
        intended_state: ServiceIntendedState,
        coreio_service_id: str | None = None,
    ) -> None:
        """Persist agent state to database."""
        self._logger.debug(
            "Persisting agent state",
            agent_id=agent_id,
            config_path=config_path,
            intended_state=intended_state.value,
            coreio_service_id=coreio_service_id,
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=0.05,
            on_backoff=lambda details: self._recover_database_and_log(agent_id, "persist agent state"),
        )
        def _do_persist_agent():
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO agent_states
                    (agent_id, config_path, intended_state, coreio_service_id, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (agent_id, config_path, intended_state.value, coreio_service_id),
                )
                conn.commit()

        _do_persist_agent()

    def update_intended_state(self, agent_id: str, intended_state: ServiceIntendedState) -> None:
        """Update only the intended state of an agent."""
        self._logger.debug(
            "Updating agent intended state",
            agent_id=agent_id,
            intended_state=intended_state.value,
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=0.05,
            on_backoff=lambda details: self._recover_database_and_log(agent_id, "update agent intended state"),
        )
        def _do_update_intended_state():
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE agent_states
                    SET intended_state = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE agent_id = ?
                    """,
                    (intended_state.value, agent_id),
                )
                conn.commit()

        _do_update_intended_state()

    def load_agents(self) -> list[dict[str, str]]:
        """Load agent data from database."""
        self._logger.info(
            "Loading agents from persistence database",
            db_path=str(self._db_path),
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=0.05,
            on_backoff=lambda details: self._recover_database_and_log(None, "load agents"),
            on_giveup=lambda details: self._logger.warning("Failed to load agents from database after recovery"),
        )
        def _do_load_agents():
            agents = []
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT agent_id, config_path, intended_state, coreio_service_id
                    FROM agent_states
                    """,
                )

                for agent_id, config_path, intended_state, coreio_service_id in cursor.fetchall():
                    self._logger.info(
                        "Loading agent from database",
                        agent_id=agent_id,
                        config_path=config_path,
                        intended_state=intended_state,
                        coreio_service_id=coreio_service_id,
                    )

                    agents.append({
                        "agent_id": agent_id,
                        "config_path": config_path,
                        "intended_state": intended_state,
                        "coreio_service_id": coreio_service_id,
                    })
            return agents

        try:
            return _do_load_agents()
        except sqlite3.DatabaseError:
            # Return empty list on database error - agents will be recreated as needed
            return []

    def remove_agent(self, agent_id: str) -> None:
        """Remove agent from persistence database."""
        self._logger.debug(
            "Removing agent from database",
            agent_id=agent_id,
        )

        @backoff.on_exception(
            backoff.constant,
            sqlite3.DatabaseError,
            max_tries=2,
            interval=0.05,
            on_backoff=lambda details: self._recover_database_and_log(agent_id, "remove agent"),
        )
        def _do_remove_agent():
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM agent_states WHERE agent_id = ?", (agent_id,))
                conn.commit()

        _do_remove_agent()
