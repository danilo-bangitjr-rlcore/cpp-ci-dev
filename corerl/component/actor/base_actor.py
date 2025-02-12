from abc import ABC, abstractmethod
from pathlib import Path

import torch

from corerl.component.policy.policy import Policy
from corerl.configs.group import Group
from corerl.state import AppState


class BaseActor(ABC):
    policy: Policy
    def __init__(self, app_state: AppState):
        self.app_state = app_state

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
    def get_log_prob(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        with_grad: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


group = Group[
    [AppState, int, int, BaseActor | None],
    BaseActor,
]()
