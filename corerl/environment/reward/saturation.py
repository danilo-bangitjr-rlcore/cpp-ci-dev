from omegaconf import DictConfig

import numpy as np
import pandas as pd

from corerl.environment.reward.base import BaseReward


class SaturationReward(BaseReward):
    def __init__(self, cfg: DictConfig):
        self.saturation_sp = 0.5

    def __call__(self, obs, **kwargs) -> float:
        reward = -np.abs(obs - self.saturation_sp).item()
        return reward
