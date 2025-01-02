import numpy as np
import pandas as pd
from typing import Any
from corerl.environment.reward.base import BaseReward


class SaturationReward(BaseReward):
    def __init__(self, cfg: Any):
        self.saturation_sp = 0.5

    def __call__(self, obs: pd.DataFrame | pd.Series, **kwargs: Any) -> float:
        reward = -np.abs(obs - self.saturation_sp).item()
        return reward
