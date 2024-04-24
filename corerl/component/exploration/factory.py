from omegaconf import DictConfig
from corerl.component.exploration.base import BaseExploration
from corerl.component.exploration.random_network import RndNetworkExplore, RndNetworkExploreLineSearch


def init_exploration_module(cfg: DictConfig, state_dim: int, action_dim: int) -> BaseExploration:
    """
    config files: root/config/agent/exploration
    """
    name = cfg.name
    if name == "random_linear":
        return RndNetworkExplore(cfg, state_dim, action_dim)
    if name == "random_linear_linesearch":
        return RndNetworkExploreLineSearch(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError