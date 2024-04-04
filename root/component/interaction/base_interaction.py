from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig
import gymnasium


class BaseInteraction(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env, state_constructor):
        self.env = env
        self.state_constructor = state_constructor

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> (np.ndarray, dict):
        raise NotImplementedError
