import numpy as np
from omegaconf import DictConfig
from pathlib import Path

import numpy
import random

from corerl.agent.base import BaseAgent
from corerl.data.data import TransitionBatch, Transition


class RandomAgent(BaseAgent):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.rng = np.random.RandomState(cfg.seed)

    def update_buffer(self, transition: Transition) -> None:
        pass

    def load_buffer(self, transitions: list[Transition]) -> None:
        pass

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        action_np = self.rng.uniform(0, 1, self.action_dim)
        
        return action_np

    def update(self) -> None:
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
