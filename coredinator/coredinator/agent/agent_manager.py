from pathlib import Path

from coredinator.agent.agent import Agent, AgentID, AgentStatus


class AgentManager:
    def __init__(self):
        self._agents: dict[AgentID, Agent] = {}

    def start_agent(self, config_path: Path):
        agent_id = AgentID(config_path.stem)
        if agent_id not in self._agents:
            self._agents[agent_id] = Agent(id=agent_id, config_path=config_path)

        self._agents[agent_id].start()
        return agent_id

    def stop_agent(self, agent_id: AgentID):
        if agent_id in self._agents:
            self._agents[agent_id].stop()

    def get_agent_status(self, agent_id: AgentID):
        if agent_id in self._agents:
            return self._agents[agent_id].status()

        return AgentStatus(id=agent_id, state="stopped", config_path=None)

    def list_agents(self):
        return list(self._agents.keys())
