import networks

def init_critic_network(cfg, input_dim, output_dim):
    """
        corresponding config : root/config/agent/critic/critic.yaml
    """
    name = cfg.name
    if name == 'ensemble':
        network = networks.EnsembleCritic(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network


def init_actor_network(cfg, input_dim, output_dim):
    if cfg.name == 'SquashedGaussian':
        network = networks.SquashedGaussian(cfg, input_dim, output_dim)
    elif cfg.name == 'Beta':
        network = networks.SquashedGaussian(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network


def init_custom_network(cfg, input_dim, output_dim):
    raise NotImplementedError