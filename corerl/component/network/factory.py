from corerl.component.network import networks
from omegaconf import DictConfig
import torch.nn as nn


def init_critic_network(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    """
    corresponding configs : corerl/config/agent/critic/critic_network
    """
    name = cfg.name
    if name == 'ensemble':
        network = networks.EnsembleCritic(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network

def init_critic_target(cfg: DictConfig, input_dim: int, output_dim: int, critic: nn.Module) -> nn.Module:
    """
    corresponding configs : corerl/config/agent/critic/critic_network
    """
    target_net = init_critic_network(cfg, input_dim, output_dim)
    target_net.load_state_dict(critic.state_dict())

    return target_net


def init_custom_network(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    name = cfg.name.lower()

    if name in ('fc', 'mlp'):
        network = networks.create_base(cfg, input_dim, output_dim)
    elif name == 'ensemble_fc':
        network = networks.EnsembleFC(cfg, input_dim, output_dim)
    elif name == 'random_linear_uncertainty':
        network = networks.RndLinearUncertainty(cfg, input_dim, output_dim)
    elif name == 'gru':
        network = networks.GRU(cfg, input_dim, output_dim)
    else:
        raise NotImplementedError

    return network
