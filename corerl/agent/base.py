from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy

from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.state import AppState

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class BaseAgentConfig:
    name: Any = MISSING

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
    def get_action_interaction(
        self,
        state: numpy.ndarray,
        prev_direct_action: numpy.ndarray,
    ) -> numpy.ndarray:  # must return a numpy array, not a tensor.
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
