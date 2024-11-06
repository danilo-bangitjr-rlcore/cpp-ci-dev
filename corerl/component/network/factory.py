import torch.nn as nn
from corerl.component.network import networks
from corerl.utils.hydra import DiscriminatedUnion, Group

critic_group = Group(
    'agent/critic/critic_network',
    return_type=nn.Module,
)

critic_group.dispatcher(networks.EnsembleCritic)


def init_critic_network(cfg: DiscriminatedUnion, input_dim: int, output_dim: int):
    return critic_group.dispatch(cfg, input_dim, output_dim)


def init_critic_target(cfg: DiscriminatedUnion, input_dim: int, output_dim: int, critic: nn.Module):
    target_net = init_critic_network(cfg, input_dim, output_dim)
    target_net.load_state_dict(critic.state_dict())

    return target_net



calibration_model_group = Group(
    'model',
    return_type=nn.Module,
)

calibration_model_group.dispatcher(networks.create_base)
calibration_model_group.dispatcher(networks.EnsembleCritic)
calibration_model_group.dispatcher(networks.RndLinearUncertainty)
calibration_model_group.dispatcher(networks.GRU)

def init_custom_network(cfg: DiscriminatedUnion, input_dim: int, output_dim: int):
    return calibration_model_group.dispatch(cfg, input_dim, output_dim)
