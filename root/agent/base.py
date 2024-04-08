from abc import ABC, abstractmethod
from omegaconf import DictConfig
import numpy


class BaseAgent(ABC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        self.replay_ratio = cfg.replay_ratio
        self.update_freq = cfg.update_freq
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.discrete_control = cfg.discrete_control
        self.seed = cfg.seed

    @abstractmethod
    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:  # must return a numpy array, not a tensor.
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_buffer(self, transition: tuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError


class BaseAC(BaseAgent):
    @abstractmethod
    def update_actor(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_critic(self) -> None:
        raise NotImplementedError
