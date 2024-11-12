import torch.nn as nn
from dataclasses import dataclass
from omegaconf import MISSING
from corerl.utils.hydra import Group


@dataclass
class BaseNetworkConfig:
    name: str = MISSING

critic_group = Group[
    [int, int],
    nn.Module,
]('agent/critic/critic_network')

custom_network_group = Group[
    [int, int],
    nn.Module,
]('model')
