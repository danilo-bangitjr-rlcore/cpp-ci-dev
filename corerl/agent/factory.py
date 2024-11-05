from corerl.agent.base import BaseAgent, group
from corerl.agent.iql import IQL
from corerl.agent.simple_ac import SimpleAC
from corerl.agent.sarsa import EpsilonGreedySarsa
from corerl.agent.inac import InAC
from corerl.agent.greedy_iql import GreedyIQL
from corerl.agent.random import RandomAgent
from corerl.utils.hydra import DiscriminatedUnion

import corerl.agent.greedy_ac # noqa: F401
import corerl.agent.action_schedule # noqa: F401


def init_agent(cfg: DiscriminatedUnion, state_dim: int, action_dim: int) -> BaseAgent:
    """
    corresponding configs: config/agent
    """

    try:
        return group.dispatch(cfg, state_dim, action_dim)
    except Exception:
        ...

    if cfg.name == 'simple_ac':
        agent = SimpleAC(cfg, state_dim, action_dim)
    elif cfg.name == 'iql':
        agent = IQL(cfg, state_dim, action_dim)
    elif cfg.name == 'inac':
        agent = InAC(cfg, state_dim, action_dim)
    elif cfg.name == 'epsilon_greedy_sarsa':
        agent = EpsilonGreedySarsa(cfg, state_dim, action_dim)
    elif cfg.name == 'greedy_iql':
        agent = GreedyIQL(cfg, state_dim, action_dim)
    elif cfg.name == 'random':
        agent = RandomAgent(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError

    return agent
