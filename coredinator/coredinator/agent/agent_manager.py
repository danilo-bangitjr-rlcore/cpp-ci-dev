import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Concatenate, cast

from coredinator.agent.agent import Agent, AgentID, AgentStatus
from coredinator.service.protocols import ServiceID, ServiceState
from coredinator.service.service_manager import ServiceManager


def db_recovery_decorator[**P, R](
    method: Callable[Concatenate["AgentManager", P], R],
) -> Callable[Concatenate["AgentManager", P], R]:
    def wrapper(self: "AgentManager", *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return method(self, *args, **kwargs)
        except sqlite3.DatabaseError:
            self._init_database()
            return method(self, *args, **kwargs)

    return cast(Any, wrapper)


class AgentManager:
    def __init__(self, base_path: Path, service_manager: ServiceManager):
        self._agents: dict[AgentID, Agent] = {}
        self._base_path = base_path
        self._db_path = base_path / "agent_state.db"
        self._service_manager = service_manager

        # Initialize database
        self._init_database()

        # Load existing agents from database and start those marked as running
        self._load_agents_from_database()

    # ----------------
    # -- Public API --
    # ----------------
    def start_agent(
        self,
        config_path: Path,
        agent_factory: type[Agent] = Agent,
        coreio_service_id: ServiceID | None = None,
    ) -> AgentID:
        agent_id = AgentID(config_path.stem)
        if agent_id not in self._agents:
            self._agents[agent_id] = agent_factory(
                id=agent_id,
                config_path=config_path,
                base_path=self._base_path,
                service_manager=self._service_manager,
                coreio_service_id=coreio_service_id,
            )

        self._agents[agent_id].start()

        # Wait a moment for processes to start, then get process IDs
        time.sleep(0.1)
        process_ids = self._agents[agent_id].get_process_ids()
        # Extract corerl and coreio process IDs (first two are always these)
        corerl_process_id = process_ids[0]
        coreio_process_id = process_ids[1]
        self._update_agent_state(agent_id, "running", config_path, corerl_process_id, coreio_process_id)

        return agent_id

    def stop_agent(self, agent_id: AgentID):
        if agent_id in self._agents:
            self._agents[agent_id].stop()

            # Update database to mark agent as stopped
            config_path = self._agents[agent_id]._config_path
            self._update_agent_state(agent_id, "stopped", config_path, None, None)

    def get_agent_status(self, agent_id: AgentID):
        if agent_id in self._agents:
            return self._agents[agent_id].status()

        return AgentStatus(id=agent_id, state=ServiceState.STOPPED, config_path=None, service_statuses={})

    def list_agents(self):
        return list(self._agents.keys())


    # -------------------
    # -- Serialization --
    # -------------------
    def _init_database(self, retries: int = 1):
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_states (
                        agent_id TEXT PRIMARY KEY,
                        operating_mode TEXT NOT NULL CHECK (operating_mode IN ('running', 'stopped')),
                        config_path TEXT NOT NULL,
                        corerl_process_id INTEGER,
                        coreio_process_id INTEGER
                    )
                """)

                conn.commit()
        except sqlite3.DatabaseError:
            if retries == 0:
                raise

            self._db_path.unlink(missing_ok=True)
            time.sleep(1)
            self._init_database(retries - 1)

    @db_recovery_decorator
    def _load_agents_from_database(self):
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT agent_id, operating_mode, config_path, corerl_process_id, coreio_process_id FROM agent_states",
            )

            for agent_id_str, operating_mode, config_path_str, corerl_process_id, coreio_process_id in (
                cursor.fetchall()
            ):
                agent_id = AgentID(agent_id_str)
                config_path = Path(config_path_str)

                # Create agent instance
                self._agents[agent_id] = Agent(
                    id=agent_id,
                    config_path=config_path,
                    base_path=self._base_path,
                    service_manager=self._service_manager,
                )

                # Start agent if it was marked as running
                if operating_mode == "running":
                    # Try to reattach to existing processes first
                    corerl_success, coreio_success = self._agents[agent_id].reattach_processes(
                        corerl_process_id,
                        coreio_process_id,
                    )

                    # If both processes were successfully reattached, no need to restart
                    if corerl_success and coreio_success:
                        # Set agent state to RUNNING when processes are successfully reattached
                        self._agents[agent_id]._agent_state = ServiceState.RUNNING
                        continue

                    # Otherwise, start the agent (which will start any missing services)
                    self._agents[agent_id].start()

    @db_recovery_decorator
    def _update_agent_state(
        self,
        agent_id: AgentID,
        operating_mode: str,
        config_path: Path,
        corerl_process_id: int | None = None,
        coreio_process_id: int | None = None,
    ):
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO agent_states
                (agent_id, operating_mode, config_path, corerl_process_id, coreio_process_id)
                VALUES (?, ?, ?, ?, ?)
            """,
                (agent_id, operating_mode, str(config_path), corerl_process_id, coreio_process_id),
            )
            conn.commit()
