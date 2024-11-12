from typing import Protocol
from omegaconf import DictConfig
import torch


class BaseExploration(Protocol):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
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
