from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

class BaseReward(ABC):
    """
    Reward class to be used in dataloader and real world environments.
    Want each reward function's __call__() method to have same signature so that they can be
    used interchangeably in the dataloader.
    """
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, obs: np.ndarray, **kwargs) -> float:
        raise NotImplementedError
