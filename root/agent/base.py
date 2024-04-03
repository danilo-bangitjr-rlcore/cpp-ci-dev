from abc import ABC, abstractmethod
import numpy

class BaseAgent(ABC):
    def __init__(self, cfg, state_dim: int, action_dim: int, discrete_control: bool):
        self.replay_ratio = cfg.replay_ratio
        self.update_freq = cfg.update_freq
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.discrete_control = discrete_control
        self.seed = cfg.seed

    @abstractmethod
    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:  # returns a numpy array, not a tensor.
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def update_buffer(self, transition: tuple):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError


class BaseAC(BaseAgent):
    @abstractmethod
    def update_actor(self):
        raise NotImplementedError

    @abstractmethod
    def update_critic(self):
        raise NotImplementedError