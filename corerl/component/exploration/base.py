from omegaconf import DictConfig
from abc import abstractmethod
import torch


class BaseExploration:
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        pass

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_exploration_bonus(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_networks(self) -> list[torch.nn.Module]:
        raise NotImplementedError
