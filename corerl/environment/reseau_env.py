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
        self.action_space = Box(cfg.action_low, cfg.action_high, shape=(1,))

    def _get_reward(self, s: np.ndarray | pd.DataFrame, a: np.ndarray):
        if self.prev_action is None:
            r = self.reward_func(df=s, prev_action=a, curr_action=a)
        else:
            r = self.reward_func(df=s, prev_action=self.prev_action, curr_action=a)
        return r

    def step(self, action: np.ndarray):
        self.take_action(action)
        end_timer = time.time() + self.obs_length
        time.sleep(end_timer - time.time())
        self.prev_action = action
        return self.get_observation(action)

    def _check_done(self) -> bool:
        return False
