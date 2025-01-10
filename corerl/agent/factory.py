from corerl.agent import agent_group, register
from corerl.agent.base import BaseAgentConfig
from corerl.state import AppState


def init_agent(cfg: BaseAgentConfig, app_state: AppState, state_dim: int, action_dim: int):
    register()
    return agent_group.dispatch(cfg, app_state, state_dim, action_dim)
