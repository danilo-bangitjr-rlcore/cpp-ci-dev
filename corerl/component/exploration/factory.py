from omegaconf import DictConfig
from root.component.exploration.random_network import BaseExploration
from root.component.exploration.random_network import RndNetworkExploreLineSearch


def init_exploration_module(cfg: DictConfig, state_dim: int, action_dim: int) -> BaseExploration:
    """
    config files: root/config/agent/exploration
    """
    name = cfg.name
    if name == "random_linear_uncertainty":
        return RndNetworkExploreLineSearch(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError