from omegaconf import DictConfig

from root.component.actor.network_actor import NetworkActor
from root.component.actor.network_actor import BaseActor


def init_actor(cfg: DictConfig, state_dim: int, action_dim: int) -> BaseActor:
    if cfg.name == 'network':
        actor = NetworkActor(cfg, state_dim, action_dim)
    else:
        raise NotImplementedError
    return actor
