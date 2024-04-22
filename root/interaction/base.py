from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig
import gymnasium

from root.state_constructor.base import BaseStateConstructor

class BaseInteraction(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env, state_constructor: BaseStateConstructor):
        self.env = env
        self.state_constructor = state_constructor

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> (np.ndarray, dict):
        raise NotImplementedError

    @abstractmethod
    def warmup_sc(self, *args, **kwargs) -> None:
        raise NotImplementedError
