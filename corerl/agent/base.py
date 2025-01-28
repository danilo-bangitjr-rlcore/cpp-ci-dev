from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy
from pydantic import Field

from corerl.component.actor.base_actor import BaseActor
from corerl.component.actor.factory import init_actor
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.critic.ensemble_critic import EnsembleCriticConfig, EnsembleQCritic, EnsembleVCritic
from corerl.component.critic.factory import init_q_critic, init_v_critic
from corerl.configs.config import MISSING, config, interpolate
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.state import AppState


@config(frozen=True)
class BaseAgentConfig:
    name: Any = MISSING

    delta_action: bool = False
    delta_bounds: tuple[float, float] | None = None
    discrete_control: bool = interpolate('${env.discrete_control}')
    freezer_freq: int = 1
    gamma: float = interpolate('${experiment.gamma}')
    n_updates: int = 1
    replay_ratio: int = 1
    seed: int = interpolate('${experiment.seed}')
    update_freq: int = 1


class BaseAgent(ABC):
    def __init__(self, cfg: BaseAgentConfig, app_state: AppState, col_desc: ColumnDescriptions):
        self.cfg = cfg
        self._app_state = app_state
        self.replay_ratio = cfg.replay_ratio
        self.update_freq = cfg.update_freq
        self.state_dim = col_desc.state_dim
        self.action_dim = col_desc.action_dim
        self.gamma = cfg.gamma
        self.discrete_control = cfg.discrete_control
        self.seed = cfg.seed
        self.n_updates = cfg.n_updates  # how many updates to apply each time update() is called
        self.freezer_freq = cfg.freezer_freq  # how often to save to freezer. This counter is not used currently.

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
        # self._app_state.event_bus.close_sync()
        return



@config(frozen=True)
class BaseACConfig(BaseAgentConfig):
    critic: EnsembleCriticConfig = Field(default_factory=EnsembleCriticConfig)
    actor: NetworkActorConfig = Field(default_factory=NetworkActorConfig)

    n_actor_updates: int = 1
    n_critic_updates: int = 1


class BaseAC(BaseAgent):
    def __init__(self, cfg: BaseACConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.n_critic_updates = cfg.n_critic_updates
        self.n_actor_updates = cfg.n_actor_updates

        self.actor: BaseActor = init_actor(cfg.actor, self.state_dim, self.action_dim)
        self.q_critic: EnsembleQCritic = init_q_critic(cfg.critic, self.state_dim, self.action_dim)
        self.v_critic: EnsembleVCritic = init_v_critic(cfg.critic, self.state_dim)

    @abstractmethod
    def update_actor(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def update_critic(self) -> list[float]:
        raise NotImplementedError
