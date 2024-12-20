from typing import Literal
import numpy as np
from pathlib import Path

from corerl.configs.config import config, list_
from corerl.data_pipeline.datatypes import NewTransition
from corerl.agent.base import BaseAgent, BaseAgentConfig


def step(start: float, end: float, step: float):
    out: list = []
    x = start
    while not np.isclose(x, end):
        out.append(x)
        x += step

    out += [end]
    return out


@config(frozen=True)
class ActionScheduleConfig(BaseAgentConfig):
    name: Literal['action_schedule'] = 'action_schedule'
    action_schedule: list[list[float]] = list_([
        step(0.5, 1.0, 0.05)
        + step(0.95, 0.0, -0.05)
        + step(0.05, 0.45, 0.05)
    ])


class ActionScheduleAgent(BaseAgent):
    def __init__(self, cfg: ActionScheduleConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.action_schedule = cfg.action_schedule
        self.step = 0

    def update_buffer(self, transition: NewTransition) -> None:
        pass

    def load_buffer(self, transitions: list[NewTransition]) -> None:
        pass

    def get_action(self, state: np.ndarray) -> np.ndarray:
        action_ind = self.step % len(self.action_schedule)
        action_np = np.array(self.action_schedule[action_ind])
        self.step += 1

        return action_np

    def update(self) -> list[float]:
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
