import torch.nn as nn
from corerl.component.network.base import critic_group, BaseNetworkConfig
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
