from typing import Any

import torch.nn as nn

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group


@config()
class BaseNetworkConfig:
    name: Any = MISSING

critic_group = Group[
    [int, int],
    nn.Module,
]()
