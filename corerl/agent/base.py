from abc import ABC, abstractmethod
from omegaconf import DictConfig
import numpy

from corerl.data.data import Transition
from corerl.utils.hooks import Hook, Hooks, When


class BaseAgent(ABC):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        self._hooks = Hooks(keys=[e.value for e in When])
        self.replay_ratio = cfg.replay_ratio
        self.update_freq = cfg.update_freq
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.discrete_control = cfg.discrete_control
        self.seed = cfg.seed
        self.n_updates = cfg.n_updates  # how many updates to apply each time update() is called
        self.freezer_freq = cfg.freezer_freq  # how often to save to freezer. This counter is not used currently.
        self.critic_buffer = None

    @abstractmethod
    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:  # must return a numpy array, not a tensor.
        """
        This method is for getting actions for the main agent/environment interaction loop.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_buffer(self, transition: Transition) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    # A function to save stats and other objects during the run
    def add_to_freezer(self) -> None:
        pass

    def register_hook(self, hook, when: When):
        self._hooks.register(hook, when)


class BaseAC(BaseAgent):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.n_critic_updates = cfg.n_critic_updates
        self.n_actor_updates = cfg.n_actor_updates

    @abstractmethod
    def update_actor(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_critic(self) -> None:
        raise NotImplementedError
