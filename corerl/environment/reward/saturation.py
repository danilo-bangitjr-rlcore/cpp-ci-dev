import numpy as np
from typing import Any
from corerl.environment.reward.base import BaseReward


class SaturationReward(BaseReward):
    def __init__(self, cfg: Any):
        self.saturation_sp = 0.5

    def __call__(self, obs, **kwargs) -> float:
        reward = -np.abs(obs - self.saturation_sp).item()
        return reward
