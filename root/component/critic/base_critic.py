from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig

class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError

    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        raise NotImplementedError


class BaseV(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: DictConfig, state_dim: int):
        super(BaseV, self).__init__(cfg)

    @abstractmethod
    def get_v(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseQ(BaseCritic):
    @abstractmethod
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super(BaseQ, self).__init__(cfg)

    @abstractmethod
    def get_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
