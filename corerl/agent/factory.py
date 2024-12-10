from corerl.agent import agent_group
from corerl.agent.base import BaseAgentConfig

import corerl.agent.greedy_iql # noqa: F401
import corerl.agent.inac # noqa: F401
import corerl.agent.simple_ac # noqa: F401
import corerl.agent.random # noqa: F401
import corerl.agent.reinforce # noqa: F401
import corerl.agent.sac # noqa: F401
import corerl.agent.iql # noqa: F401
import corerl.agent.greedy_ac # noqa: F401
import corerl.agent.action_schedule # noqa: F401
import corerl.agent.sarsa # noqa: F401


def init_agent(cfg: BaseAgentConfig, state_dim: int, action_dim: int):
    return agent_group.dispatch(cfg, state_dim, action_dim)
