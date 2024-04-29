from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig
import gymnasium

from corerl.state_constructor.base import BaseStateConstructor

class BaseInteraction(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env, state_constructor: BaseStateConstructor):
        self.env = env
        self.state_constructor = state_constructor

    @abstractmethod
    # step returns a list of transitions and a list of environment infos
    # the reason it takes in the current state is to build these transitions
    def step(self, state: np.ndarray,  action: np.ndarray) -> tuple[list[tuple], list[dict]]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> (np.ndarray, dict):
        raise NotImplementedError

    @abstractmethod
    def warmup_sc(self, *args, **kwargs) -> None:
        raise NotImplementedError


    def get_state_dim(self) -> int:
        obs, _ = self.env.reset()
        return self.state_constructor.get_state_dim(obs)
