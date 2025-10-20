from abc import ABC, abstractmethod
from pathlib import Path

import numpy
from lib_agent.buffer.buffer import State

from corerl.configs.agent.base import BaseAgentConfig
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.state import AppState


class BaseAgent(ABC):
    def __init__(self, cfg: BaseAgentConfig, app_state: AppState, col_desc: ColumnDescriptions):
        self.cfg = cfg
        self._app_state = app_state
        self.state_dim = col_desc.state_dim
        self.action_dim = col_desc.action_dim
        self.gamma = cfg.gamma
        self.seed = cfg.seed
        self.n_updates = cfg.n_updates  # how many updates to apply each time update() is called

    @abstractmethod
    def get_action_interaction(self, state: State) -> numpy.ndarray: # must return a numpy array, not a jax Array
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
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError

    def get_buffer_sizes(self) -> dict[str, list[int]]:
        return {}

    def close(self):
        return
