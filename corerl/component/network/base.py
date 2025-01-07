import torch.nn as nn
from corerl.configs.config import config
from corerl.configs.config import MISSING
from corerl.configs.group import Group


@config(frozen=True)
class BaseNetworkConfig:
    name: str = MISSING

critic_group = Group[
    [int, int],
    nn.Module,
]()
