import torch
from typing import Protocol
from corerl.utils.hydra import DiscriminatedUnion, Group

class BaseExploration(Protocol):
    def __init__(self, cfg: DiscriminatedUnion, state_dim: int, action_dim: int):
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
]('agent/exploration')
