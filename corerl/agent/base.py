from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import field
from typing import Any
import numpy
from pathlib import Path

from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.critic.ensemble_critic import EnsembleCriticConfig
from corerl.configs.config import MISSING, interpolate, config
from corerl.data_pipeline.datatypes import NewTransition
from corerl.utils.hook import Hooks, when
from corerl.messages.client import MessageBusClientConfig, make_msg_bus_client


@config(frozen=True)
class BaseAgentConfig:
    name: Any = MISSING

    discrete_control: bool = interpolate('${env.discrete_control}')
    freezer_freq: int = 1
    gamma: float = interpolate('${experiment.gamma}')
    message_bus: MessageBusClientConfig = field(default_factory=MessageBusClientConfig)
    n_updates: int = 1
    replay_ratio: int = 1
    seed: int = interpolate('${experiment.seed}')
    update_freq: int = 1


class BaseAgent(ABC):
    def __init__(self, cfg: BaseAgentConfig, state_dim: int, action_dim: int):
        self._hooks = Hooks(keys=[e.value for e in when.Agent])
        self.replay_ratio = cfg.replay_ratio
        self.update_freq = cfg.update_freq
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.discrete_control = cfg.discrete_control
        self.seed = cfg.seed
        self.n_updates = cfg.n_updates  # how many updates to apply each time update() is called
        self.freezer_freq = cfg.freezer_freq  # how often to save to freezer. This counter is not used currently.

        self._msg_bus = make_msg_bus_client(cfg.message_bus)
        self._msg_bus.start_sync()

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
    def update_buffer(self, transitions: Sequence[NewTransition]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_buffer(self, transitions: Sequence[NewTransition]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError

    def register_hook(self, hook, when: when.Agent):
        self._hooks.register(hook, when)

    def get_buffer_sizes(self) -> dict[str, list[int]]:
        return {}

    def close(self):
        self._msg_bus.close_sync()



@config(frozen=True)
class BaseACConfig(BaseAgentConfig):
    critic: EnsembleCriticConfig = field(default_factory=EnsembleCriticConfig)
    actor: NetworkActorConfig = field(default_factory=NetworkActorConfig)

    n_actor_updates: int = 1
    n_critic_updates: int = 1


class BaseAC(BaseAgent):
    def __init__(self, cfg: BaseACConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.n_critic_updates = cfg.n_critic_updates
        self.n_actor_updates = cfg.n_actor_updates

    @abstractmethod
    def update_actor(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def update_critic(self) -> None:
        raise NotImplementedError
