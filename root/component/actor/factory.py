from omegaconf import DictConfig

from root.component.actor.network_actor import NetworkActor
from root.component.actor.network_actor import BaseActor


def init_actor(cfg: DictConfig, state_dim: int, action_dim: int, initializer: BaseActor | None = None) -> BaseActor:
    """
    corresponding configs: config/agent/actor
    """
    if cfg.name == 'network':
        actor = NetworkActor(cfg, state_dim, action_dim, initializer=initializer)
    else:
        raise NotImplementedError
    return actor
