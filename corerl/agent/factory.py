from corerl.agent.base import BaseAgent, group
from corerl.agent.inac import InAC
from corerl.agent.greedy_iql import GreedyIQL
from corerl.utils.hydra import DiscriminatedUnion

import corerl.agent.simple_ac # noqa: F401
import corerl.agent.random # noqa: F401
import corerl.agent.reinforce # noqa: F401
import corerl.agent.sac # noqa: F401
import corerl.agent.iql # noqa: F401
import corerl.agent.greedy_ac # noqa: F401
import corerl.agent.action_schedule # noqa: F401
import corerl.agent.sarsa # noqa: F401


def init_agent(cfg: DiscriminatedUnion, state_dim: int, action_dim: int) -> BaseAgent:
    """
    corresponding configs: config/agent
    """

    try:
        return group.dispatch(cfg, state_dim, action_dim)
    except Exception:
        ...

    if cfg.name == 'inac':
        agent = InAC(cfg, state_dim, action_dim)
    elif cfg.name == 'greedy_iql':
        agent = GreedyIQL(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError

    return agent
