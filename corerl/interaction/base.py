from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig
import gymnasium

from corerl.state_constructor.base import BaseStateConstructor
from corerl.data import Transition


class BaseInteraction(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig, env: gymnasium.Env, state_constructor: BaseStateConstructor):
        self.env = env
        self.state_constructor = state_constructor
        self.timeout = cfg.timeout  # When timeout is set to 0, there is no timeout.
        self.internal_clock = 0

    @abstractmethod
    # step returns a list of transitions and a list of environment infos
    def step(self, action: np.ndarray) -> tuple[list[Transition], list[dict]]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> (np.ndarray, dict):
        raise NotImplementedError

    @abstractmethod
    def warmup_sc(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def env_counter(self):
        self.internal_clock += 1
        trunc = (self.timeout > 0) and (self.internal_clock % self.timeout == 0)
        return trunc
