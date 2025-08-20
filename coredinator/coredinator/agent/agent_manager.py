from coredinator.agent.agent_process import AgentID, AgentProcess


class AgentManager:
    def __init__(self):
        self.agents: dict[AgentID, AgentProcess] = {}

    def start_agent(self, config_id: str):
        ...

    def stop_agent(self, config_id: str):
        ...

    def get_agent_status(self, config_id: str):
        ...

    def list_agents(self):
        return list(self.agents.keys())
