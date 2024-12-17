import torch
from typing import Any, Protocol
from corerl.configs.config import config
from corerl.configs.config import MISSING
from corerl.configs.group import Group


@config()
class BaseExplorationConfig:
    name: Any = MISSING


class BaseExploration(Protocol):
    def __init__(self, cfg: BaseExplorationConfig, state_dim: int, action_dim: int):
        ...

    def update(self) -> None:
        ...

    def get_exploration_bonus(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def get_networks(self) -> list[torch.nn.Module]:
        ...


explore_group = Group[
    [int, int],
    BaseExploration,
]()
