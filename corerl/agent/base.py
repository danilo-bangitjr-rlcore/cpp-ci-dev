from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy
from pydantic import Field

from corerl.component.actor.base_actor import BaseActor
from corerl.component.actor.factory import init_actor
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.critic.base_critic import CriticConfig, EnsembleCritic
from corerl.component.critic.factory import init_q_critic
from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.state import AppState

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class BaseAgentConfig:
    name: Any = MISSING

    delta_action: bool = False
    delta_bounds: list[tuple[float, float]] = Field(default_factory=list)
    n_updates: int = 1
    replay_ratio: int = 1
    update_freq: int = 1

    gamma: float = MISSING
    seed: int = MISSING

    @computed('gamma')
    @classmethod
    def _gamma(cls, cfg: 'MainConfig'):
        return cfg.experiment.gamma


    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.experiment.seed


class BaseAgent(ABC):
    def __init__(self, cfg: BaseAgentConfig, app_state: AppState, col_desc: ColumnDescriptions):
        self.cfg = cfg
        self._app_state = app_state
        self.replay_ratio = cfg.replay_ratio
        self.update_freq = cfg.update_freq
        self.state_dim = col_desc.state_dim
        self.action_dim = col_desc.action_dim
        self.gamma = cfg.gamma
        self.seed = cfg.seed
        self.n_updates = cfg.n_updates  # how many updates to apply each time update() is called

    @abstractmethod
    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:  # must return a numpy array, not a tensor.
        """
        This method is for getting actions for the main agent/environment interaction loop.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def update_buffer(self, pr: PipelineReturn) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_buffer(self, pr: PipelineReturn) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError

    def get_buffer_sizes(self) -> dict[str, list[int]]:
        return {}

    def close(self):
        return



@config()
class BaseACConfig(BaseAgentConfig):
    critic: CriticConfig = Field(default_factory=CriticConfig)
    actor: NetworkActorConfig = Field(default_factory=NetworkActorConfig)

    n_actor_updates: int = 1
    n_critic_updates: int = 1


class BaseAC(BaseAgent):
    def __init__(self, cfg: BaseACConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.n_critic_updates = cfg.n_critic_updates
        self.n_actor_updates = cfg.n_actor_updates

        # the implicit action dim is doubled when using delta actions
        # because we will be receiving both the direct action and the
        # delta action in each transition
        if self.cfg.delta_action:
            self.action_dim = int(self.action_dim / 2)

        self.actor: BaseActor = init_actor(cfg.actor, app_state, self.state_dim, self.action_dim)
        self.critic: EnsembleCritic = init_q_critic(cfg.critic, app_state, self.state_dim, self.action_dim)

    @abstractmethod
    def update_actor(self) -> object:
        raise NotImplementedError

    @abstractmethod
    def update_critic(self) -> list[float]:
        raise NotImplementedError
