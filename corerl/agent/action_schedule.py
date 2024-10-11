import numpy as np
from omegaconf import DictConfig
from pathlib import Path

import numpy

from corerl.agent.base import BaseAgent
from corerl.data.data import Transition


class ActionScheduleAgent(BaseAgent):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.action_schedule = cfg.action_schedule
        self.step = 0

    def update_buffer(self, transition: Transition) -> None:
        pass

    def load_buffer(self, transitions: list[Transition]) -> None:
        pass

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        action_ind = self.step % len(self.action_schedule)
        action_np = np.array(self.action_schedule[action_ind])
        self.step += 1
        
        return action_np

    def update(self) -> None:
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
