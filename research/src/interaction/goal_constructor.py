from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class TagConfig:
    name: str
    operating_range: tuple[float, float]
    red_bounds: tuple[float, float] | None = None
    yellow_bounds: tuple[float, float] | None = None


@dataclass
class Goal:
    tag: str
    op: Literal['up_to', 'down_to']
    thresh: float


@dataclass
class RewardConfig:
    priorities: list[Goal]


class GoalConstructor:
    def __init__(self, reward_cfg: RewardConfig, tag_cfgs: list[TagConfig]):
        self._cfg = reward_cfg
        self._tag_cfgs = tag_cfgs

    def __call__(self, observation: np.ndarray) -> float:
        df = pd.DataFrame([observation], columns=[f'tag-{i}' for i in range(len(observation))])

        for priority in self._cfg.priorities:
            tag_value = df[priority.tag].iloc[0]
            tag_cfg = next(t for t in self._tag_cfgs if t.name == priority.tag)

            # calculate violation percentage
            if priority.op == 'down_to':
                if tag_value > priority.thresh:
                    violation = (tag_value - priority.thresh) / (tag_cfg.operating_range[1] - priority.thresh)
                    return -violation
            else:  # up_to
                if tag_value < priority.thresh:
                    violation = (priority.thresh - tag_value) / (priority.thresh - tag_cfg.operating_range[0])
                    return -violation

        return 0.0
