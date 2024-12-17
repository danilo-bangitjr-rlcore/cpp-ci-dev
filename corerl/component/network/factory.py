import torch.nn as nn
from corerl.component.network.base import critic_group, custom_network_group, BaseNetworkConfig
from corerl.component.network import networks

NetworkConfig = (
    networks.EnsembleCriticNetworkConfig
)

critic_group.dispatcher(networks.EnsembleCritic)


def init_critic_network(cfg: BaseNetworkConfig, input_dim: int, output_dim: int):
    return critic_group.dispatch(cfg, input_dim, output_dim)


def init_critic_target(cfg: BaseNetworkConfig, input_dim: int, output_dim: int, critic: nn.Module):
    target_net = init_critic_network(cfg, input_dim, output_dim)
    target_net.load_state_dict(critic.state_dict())

    return target_net


CustomNetworkConfig = (
    networks.NNTorsoConfig
    | networks.EnsembleCriticNetworkConfig
    | networks.RndLinearUncertaintyConfig
    | networks.GRUConfig
)

custom_network_group.dispatcher(networks.create_base)
custom_network_group.dispatcher(networks.EnsembleCritic)
custom_network_group.dispatcher(networks.RndLinearUncertainty)
custom_network_group.dispatcher(networks.GRU)

def init_custom_network(cfg: BaseNetworkConfig, input_dim: int, output_dim: int):
    return custom_network_group.dispatch(cfg, input_dim, output_dim)
