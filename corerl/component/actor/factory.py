from omegaconf import DictConfig

from corerl.component.actor.network_actor import NetworkActor, NetworkActorLineSearch


def init_actor(
    cfg: DictConfig,
    state_dim: int,
    action_dim: int,
    initializer: NetworkActor | None = None,
) -> NetworkActor:
    """
    corresponding configs: config/agent/actor
    """
    if cfg.name == 'network':
        actor = NetworkActor(cfg, state_dim, action_dim, initializer=initializer)
    elif cfg.name == 'network_linesearch':
        actor = NetworkActorLineSearch(cfg, state_dim, action_dim, initializer=initializer)
    else:
        raise NotImplementedError
    return actor
