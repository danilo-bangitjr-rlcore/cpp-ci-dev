from root.component.network import networks
from omegaconf import DictConfig
import torch.nn as nn


def init_critic_network(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
        corresponding config : root/config/agent/critic/critic.yaml
    """
    name = cfg.name
    if name == 'ensemble':
        network = networks.EnsembleCritic(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network


def init_actor_network(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
        corresponding config : root/config/agent/actor/network.yaml
    """

    if cfg.name == 'SquashedGaussian':
        network = networks.SquashedGaussian(cfg, input_dim, output_dim)
    elif cfg.name == 'Beta':
        network = networks.BetaPolicy(cfg, input_dim, output_dim)
    elif cfg.name == 'Softmax':
        network = networks.Softmax(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network


# TODO : implement this
def init_custom_network(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    raise NotImplementedError
