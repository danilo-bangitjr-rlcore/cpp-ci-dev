import torch

from pathlib import Path
from abc import ABC, abstractmethod

from corerl.utils.hydra import Group


class BaseActor(ABC):
    @abstractmethod
    def get_action(
        self,
        state: torch.Tensor,
        with_grad: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        loss: torch.Tensor,
        opt_args: tuple = tuple(),
        opt_kwargs: dict | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


group = Group(
    'agent/actor',
    return_type=BaseActor,
)
