from abc import ABC, abstractmethod
from pathlib import Path

import torch

from corerl.configs.config import config
from corerl.state import AppState


@config()
class BaseCriticConfig:
    ...


class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg: BaseCriticConfig,  app_state: AppState):
        self.app_state = app_state

    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


class BaseQ(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: BaseCriticConfig, app_state: AppState, state_dim: int, action_dim: int):
        super(BaseQ, self).__init__(cfg, app_state)

    @abstractmethod
    def get_q(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_q_target(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError
