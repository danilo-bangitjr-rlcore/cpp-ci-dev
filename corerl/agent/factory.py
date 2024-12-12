from corerl.agent import agent_group, register
from corerl.agent.base import BaseAgentConfig


def init_agent(cfg: BaseAgentConfig, state_dim: int, action_dim: int):
    register()
    return agent_group.dispatch(cfg, state_dim, action_dim)
