from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy
import numpy as np

from corerl.agent.base import BaseAgent, BaseAgentConfig
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.pipeline import ColumnDescriptions
from corerl.state import AppState


@config(frozen=True)
class RandomAgentConfig(BaseAgentConfig):
    name: Literal['random'] = 'random'

class RandomAgent(BaseAgent):
    def __init__(self, cfg: RandomAgentConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.rng = np.random.RandomState(cfg.seed)

    def update_buffer(self, transitions: Sequence[Transition]) -> None:
        pass

    def load_buffer(self, transitions: Sequence[Transition]) -> None:
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
