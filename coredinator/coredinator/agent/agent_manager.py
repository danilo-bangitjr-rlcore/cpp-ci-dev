from pathlib import Path

from coredinator.agent.agent_process import AgentID, AgentProcess, AgentStatus


class AgentManager:
    def __init__(self):
        self.agents: dict[AgentID, AgentProcess] = {}

    def start_agent(self, config_path: Path):
        agent_id = AgentID(config_path.stem)
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentProcess(agent_id, config_path)

        self.agents[agent_id].start()
        return agent_id

    def stop_agent(self, agent_id: AgentID):
        if agent_id in self.agents:
            self.agents[agent_id].stop()

    def get_agent_status(self, agent_id: AgentID):
        if agent_id not in self.agents:
            return AgentStatus(
                id=agent_id,
                state="stopped",
                config_path=None,
            )

        return self.agents[agent_id].status()

    def list_agents(self):
        return list(self.agents.keys())
