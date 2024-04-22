from omegaconf import DictConfig
from corerl.agent.base import BaseAgent
from corerl.agent.iql import IQL
from corerl.agent.simple_ac import SimpleAC
from corerl.agent.sarsa import EpsilonGreedySarsa
from corerl.agent.inac import InAC
from corerl.agent.greedy_ac import GreedyAC
from corerl.agent.greedy_iql import GreedyIQL


def init_agent(cfg: DictConfig, state_dim: int, action_dim: int) -> BaseAgent:
    """
    corresponding configs: config/agent
    """
    if cfg.name == 'simple_ac':
        agent = SimpleAC(cfg, state_dim, action_dim)
    elif cfg.name == 'iql':
        agent = IQL(cfg, state_dim, action_dim)
    elif cfg.name == 'inac':
        agent = InAC(cfg, state_dim, action_dim)
    elif cfg.name == 'epsilon_greedy_sarsa':
        agent = EpsilonGreedySarsa(cfg, state_dim, action_dim)
    elif cfg.name == 'greedy_ac':
        agent = GreedyAC(cfg, state_dim, action_dim)
    elif cfg.name == 'greedy_iql':
        agent = GreedyIQL(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError

    return agent
