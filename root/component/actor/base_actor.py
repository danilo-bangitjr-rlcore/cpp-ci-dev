import torch
import numpy as np

from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseActor(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, state: np.ndarray) -> (np.ndarray | torch.Tensor, dict):
        raise NotImplementedError
