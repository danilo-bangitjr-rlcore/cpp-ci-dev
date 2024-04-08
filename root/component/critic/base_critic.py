from abc import ABC, abstractmethod

import numpy as np
import torch
from omegaconf import DictConfig


class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        print(cfg.discrete_control)
        assert False

        self.discrete_control = cfg.discrete_control

    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError


class BaseV(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: DictConfig, state_dim: int):
        super(BaseV, self).__init__(cfg)

    @abstractmethod
    def get_v(self, state: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor | np.ndarray:
        raise NotImplementedError


class BaseQ(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super(BaseQ, self).__init__(cfg)

    @abstractmethod
    def get_q(self, state: torch.Tensor | np.ndarray, action: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor | np.ndarray:
        raise NotImplementedError
