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
    op: Literal['up_to', 'down_to', 'min', 'max']
    thresh: float


@dataclass
class RewardConfig:
    priorities: list[Goal]


class GoalConstructor:
    def __init__(self, reward_cfg: RewardConfig, tag_cfgs: list[TagConfig]):
        self._cfg = reward_cfg
        self._tag_cfgs = {cfg.name: cfg for cfg in tag_cfgs}

    def _evaluate_threshold(self, thresh: str | float, df: pd.DataFrame) -> float:
        if not isinstance(thresh, str) or '{' not in thresh:
            return float(thresh)

        for i in range(len(df.columns)):
            thresh = thresh.replace(f'{{tag-{i}}}', f'df["tag-{i}"].iloc[0]')
        return eval(thresh)

    def _calculate_violation(self, tag_value: float, thresh: float,
                           op_range: tuple[float, float], op: str) -> float:
        if op == 'down_to' and tag_value > thresh:
            return (tag_value - thresh) / (op_range[1] - thresh)
        if op == 'up_to' and tag_value < thresh:
            return (thresh - tag_value) / (thresh - op_range[0])
        return 0.0

    def _normalize_optimization_value(self, value: float, op_range: tuple[float, float],
                                    op: str) -> float:
        normalized = (value - op_range[0]) / (op_range[1] - op_range[0])
        return 1.0 - normalized if op == 'min' else normalized

    def __call__(self, observation: np.ndarray) -> float:
        df = pd.DataFrame([observation], columns=pd.Index([f'tag-{i}' for i in range(len(observation))]))

        # calculate bucket size for non-optimization priorities
        non_opt_priorities = [p for p in self._cfg.priorities if p.op in ['up_to', 'down_to']]
        bucket_size = 0.5 / len(non_opt_priorities) if non_opt_priorities else 0

        for i, priority in enumerate(self._cfg.priorities):
            tag_value = df[priority.tag].iloc[0]
            tag_cfg = self._tag_cfgs[priority.tag]

            if priority.op in ['up_to', 'down_to']:
                thresh = self._evaluate_threshold(priority.thresh, df)
                violation = self._calculate_violation(tag_value, thresh,
                                                   tag_cfg.operating_range, priority.op)
                if violation > 0:
                    return -1 + (i * bucket_size) + (violation * bucket_size)

            elif priority.op in ['min', 'max']:
                opt_values = []
                for opt_priority in self._cfg.priorities[i:]:
                    if opt_priority.op not in ['min', 'max']:
                        break
                    opt_value = df[opt_priority.tag].iloc[0]
                    opt_cfg = self._tag_cfgs[opt_priority.tag]
                    opt_values.append(self._normalize_optimization_value(
                        opt_value, opt_cfg.operating_range, opt_priority.op))

                if opt_values:
                    return -0.5 + (sum(opt_values) / len(opt_values) * 0.5)
        # all priorities satisfied then zero rewards
        return 0.0
