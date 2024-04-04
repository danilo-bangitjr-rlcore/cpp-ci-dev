from root.agent.simple_ac import SimpleAC
from omegaconf import DictConfig
from root.agent.base import BaseAgent
from typing import Type


def init_agent(cfg: DictConfig, state_dim: int, action_dim: int) -> BaseAgent:
    if cfg.name == 'SimpleAC':
        agent = SimpleAC(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError

    return agent
