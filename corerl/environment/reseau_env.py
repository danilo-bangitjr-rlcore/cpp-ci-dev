import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from corerl.configs.config import MISSING, config, list_
from corerl.environment.influx_opc_env import InfluxOPCConfig, InfluxOPCEnv
from corerl.environment.reward.factory import init_reward_function
from corerl.environment.utils.cast_configs import cast_dict_to_config


@config()
class ReseauConfig(InfluxOPCConfig):
    reward: Any = MISSING
    action_low: float = MISSING
    action_high: float = MISSING
    obs_space_low: list[float] = list_()
    obs_space_high: list[float] = list_()
    col_names: list[str] = list_()
    action_names: list[str] = list_()
    endo_obs_names: list[str] = list_()
    endo_inds: list[int] = list_()


class ReseauEnv(InfluxOPCEnv):
    def __init__(self, cfg: dict | ReseauConfig):
        if isinstance(cfg, dict):
            cfg = cast_dict_to_config(cfg, ReseauConfig)
        super().__init__(cfg)
        self.reward_func = init_reward_function(cfg.reward)
        self.prev_action = None
        self.action_space = Box(low=cfg.action_low, high=cfg.action_high, shape=(1,))
        self.observation_space = Box(low=np.array(cfg.obs_space_low), high=np.array(cfg.obs_space_high))
        self.col_names = cfg.col_names
        self.action_names = cfg.action_names
        self.obs_col_names = cfg.obs_col_names
        self.endo_obs_names = cfg.endo_obs_names
        self.endo_inds = cfg.endo_inds

    def _get_reward(
        self,
        obs: np.ndarray,
        obs_series: pd.Series,
        steps_until_decision: int | None,
        action: np.ndarray,
        decision_point: bool,
    ):
        if self.prev_action is None:
            r = self.reward_func(obs_series, prev_action=action, curr_action=action)
        else:
            r = self.reward_func(obs_series, prev_action=self.prev_action, curr_action=action)
            self.prev_action = action
        return r

    def step(self, action: np.ndarray):
        self.take_action(action)
        end_timer = time.time() + self.obs_length.total_seconds()
        time.sleep(end_timer - time.time())
        self.prev_action = action
        return self.get_observation(action) # type: ignore

    def _check_done(self) -> bool:
        return False

    async def get_deployed_action(self, time: datetime | None = None) -> np.ndarray:
        ...
