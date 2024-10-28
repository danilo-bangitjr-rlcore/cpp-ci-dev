from datetime import datetime
import numpy as np
import pandas as pd
import time
from gymnasium.spaces import Box

from corerl.environment.reward.factory import init_reward_function
from corerl.environment.influx_opc_env import InfluxOPCEnv


class ReseauEnv(InfluxOPCEnv):
    def __init__(self, cfg):
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

    def _get_reward(self, s: np.ndarray | pd.DataFrame, a: np.ndarray):
        assert isinstance(s, np.ndarray)
        if self.prev_action is None:
            r = self.reward_func(s, prev_action=a, curr_action=a)
        else:
            r = self.reward_func(s, prev_action=self.prev_action, curr_action=a)
            self.prev_action = a
        return r

    def step(self, action: np.ndarray):
        self.take_action(action)
        end_timer = time.time() + self.obs_length
        time.sleep(end_timer - time.time())
        self.prev_action = action
        return self.get_observation(action)

    def _check_done(self) -> bool:
        return False

    async def healthcheck(self):
        ...

    async def async_reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        ...

    async def get_deployed_action(self, time: datetime | None = None) -> np.ndarray:
        ...
