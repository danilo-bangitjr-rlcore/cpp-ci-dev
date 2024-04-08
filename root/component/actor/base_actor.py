import torch
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseActor(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, state: np.ndarray | torch.Tensor, **kwargs) -> (np.ndarray | torch.Tensor, dict):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError