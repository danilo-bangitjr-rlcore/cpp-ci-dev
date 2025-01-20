import torch.nn as nn

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group


@config(frozen=True)
class BaseNetworkConfig:
    name: str = MISSING

critic_group = Group[
    [int, int],
    nn.Module,
]()
