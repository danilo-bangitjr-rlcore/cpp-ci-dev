from omegaconf import DictConfig

import numpy as np
import pandas as pd

from corerl.environment.reward.base import BaseReward

class ReseauReward(BaseReward):
    def __init__(self, cfg: DictConfig):
        self.orp_sp = cfg.orp_sp
        self.orp_col_name = cfg.orp_col_name
        self.penalty_weight = cfg.penalty_weight
        self.action_scale = cfg.action_scale

    def __call__(self, **kwargs) -> float:
        df = kwargs['df']
        prev_action = kwargs['prev_action']
        curr_action = kwargs['curr_action']
        
        orps = df[self.orp_col_name].to_numpy()
        orps = orps[~np.isnan(orps)]

        mae = np.mean(np.abs(self.orp_sp - orps))

        if prev_action == None:
            penalty = 0.0
        else:
            penalty = self.penalty_weight * abs(prev_action - curr_action) / self.action_scale

        if mae <= 5.0:
            return np.clip(-np.log(0.2*mae), 0.0, 5.0) - penalty
        else:
            return -0.2*mae + 1.0 - penalty