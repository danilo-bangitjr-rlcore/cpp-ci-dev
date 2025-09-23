from pathlib import Path

from coredinator.agent.agent import Agent, AgentID, AgentStatus
from coredinator.agent.persistence import AgentPersistenceLayer
from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceID, ServiceIntendedState, ServiceState
from coredinator.service.service_manager import ServiceManager


class AgentManager:
    def __init__(self, base_path: Path, service_manager: ServiceManager):
        self._logger = get_logger(__name__)
        self._agents: dict[AgentID, Agent] = {}
        self._base_path = base_path
        self._service_manager = service_manager

        # Initialize agent persistence
        self._agent_persistence = AgentPersistenceLayer(base_path / "agent_state.db")

        # Load persisted agents on startup
        self.load_agents()

        self._logger.info(
            "Initializing AgentManager",
            base_path=str(base_path),
        )

    def load_agents(self) -> None:
        """Load and restore persisted agents on startup."""
        self._logger.info("Loading persisted agents")

        persisted_agents = self._agent_persistence.load_agents()
        for agent_data in persisted_agents:
            agent_id = AgentID(agent_data["agent_id"])

            self._logger.info(
                "Restoring persisted agent",
                agent_id=agent_id,
                config_path=agent_data["config_path"],
                intended_state=agent_data["intended_state"],
            )

            # Create agent instance
            coreio_service_id = (
                ServiceID(agent_data["coreio_service_id"])
                if agent_data["coreio_service_id"]
                else None
            )
            self._agents[agent_id] = Agent(
                id=agent_id,
                config_path=Path(agent_data["config_path"]),
                base_path=self._base_path,
                service_manager=self._service_manager,
                coreio_service_id=coreio_service_id,
            )

            # If intended state is RUNNING, start the agent
            if agent_data["intended_state"] == ServiceIntendedState.RUNNING.value:
                self._agents[agent_id].start()

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

        self._logger.info(
            "Starting agent",
            agent_id=agent_id,
            config_path=str(config_path),
            coreio_service_id=coreio_service_id,
        )

        if agent_id not in self._agents:
            self._agents[agent_id] = agent_factory(
                id=agent_id,
                config_path=config_path,
                base_path=self._base_path,
                service_manager=self._service_manager,
                coreio_service_id=coreio_service_id,
            )

        self._agents[agent_id].start()

        # Persist agent state
        self._agent_persistence.persist_agent(
            agent_id=str(agent_id),
            config_path=str(config_path),
            intended_state=ServiceIntendedState.RUNNING,
            coreio_service_id=str(coreio_service_id) if coreio_service_id else None,
        )

        self._logger.info(
            "Agent started successfully",
            agent_id=agent_id,
        )

        return agent_id

    def stop_agent(self, agent_id: AgentID):
        self._logger.info(
            "Stopping agent",
            agent_id=agent_id,
        )

        if agent_id in self._agents:
            self._agents[agent_id].stop()

            # Persist agent state
            self._agent_persistence.update_intended_state(str(agent_id), ServiceIntendedState.STOPPED)

            self._logger.info(
                "Agent stopped successfully",
                agent_id=agent_id,
            )
        else:
            self._logger.warning(
                "Attempted to stop non-existent agent",
                agent_id=agent_id,
            )

    def get_agent_status(self, agent_id: AgentID):
        if agent_id in self._agents:
            return self._agents[agent_id].status()

        return AgentStatus(id=agent_id, state=ServiceState.STOPPED, config_path=None, service_statuses={})

    def list_agents(self):
        return list(self._agents.keys())
