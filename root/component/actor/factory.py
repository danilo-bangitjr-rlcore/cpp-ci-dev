from network_actor import NetworkActor


def init_actor(cfg, state_dim, action_dim):
    if cfg.name == 'network':
        actor = NetworkActor(cfg,  state_dim, action_dim)
    else:
        raise NotImplementedError
    return actor