import numpy as np
import torch
from omegaconf import DictConfig

from pathlib import Path
from abc import ABC, abstractmethod

from corerl.utils.types import TensorLike


class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        self.discrete_control = cfg.discrete_control

    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


class BaseV(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: DictConfig, state_dim: int):
        super(BaseV, self).__init__(cfg)

    @abstractmethod
    def get_v(self, state: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor | np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_v_target(self, state: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor | np.ndarray:
        raise NotImplementedError


class BaseQ(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super(BaseQ, self).__init__(cfg)

    @abstractmethod
    def get_q(self, state: TensorLike, action: TensorLike,
              **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_q_target(self, state: TensorLike, action: TensorLike,
                     **kwargs) -> torch.Tensor:
        raise NotImplementedError
