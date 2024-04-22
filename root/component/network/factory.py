from root.component.network import networks
from omegaconf import DictConfig
import torch.nn as nn


def init_critic_network(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
    corresponding configs : root/config/agent/critic/critic_network
    """
    name = cfg.name
    if name == 'ensemble':
        network = networks.EnsembleCritic(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network


def init_actor_network(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
    corresponding configs : root/config/agent/actor/actor_network
    """
    if cfg.name == 'squashed_gaussian':
        network = networks.SquashedGaussian(cfg, input_dim, output_dim)
    elif cfg.name == 'beta':
        network = networks.BetaPolicy(cfg, input_dim, output_dim)
    elif cfg.name == 'softmax':
        network = networks.Softmax(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network


def init_custom_network(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    if cfg.name == 'fc':
        network = networks.FC(cfg, input_dim, output_dim)
    elif cfg.name == 'random_linear_uncertainty':
        network = networks.RndLinearUncertainty(cfg, input_dim, output_dim)
    elif cfg.name == 'random_beta':
        if cfg.discrete_control:
            network = networks.UniformRandomDisc(cfg, input_dim, output_dim)
        else:
            network = networks.UniformRandomCont(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network
