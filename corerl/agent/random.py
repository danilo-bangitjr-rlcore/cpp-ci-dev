from typing import Literal
import numpy as np
from pathlib import Path

import numpy

from corerl.configs.config import config
from corerl.agent.base import BaseAgent, BaseAgentConfig
from corerl.data_pipeline.datatypes import NewTransition


@config(frozen=True)
class RandomAgentConfig(BaseAgentConfig):
    name: Literal['random'] = 'random'

class RandomAgent(BaseAgent):
    def __init__(self, cfg: RandomAgentConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.rng = np.random.RandomState(cfg.seed)

    def update_buffer(self, transition: NewTransition) -> None:
        pass

    def load_buffer(self, transitions: list[NewTransition]) -> None:
        pass

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        action_np = self.rng.uniform(0, 1, self.action_dim)

        return action_np

    def update(self) -> list[float]:
        return []

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
