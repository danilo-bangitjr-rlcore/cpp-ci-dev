from abc import ABC, abstractmethod
from omegaconf import DictConfig

import gymnasium
import numpy as np


class BaseStateConstructor(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env):
        raise NotImplementedError

    @abstractmethod
    def update(self, observation: np.ndarray, **kwargs):
        raise NotImplementedError
