from typing import Any, Literal

import gymnasium as gym
import numpy as np
from lib_config.config import config

from rl_env.group_util import EnvConfig, env_group


@config(frozen=True)
class MountainCarContinuousConfig(EnvConfig):
    name: Literal['MountainCarContinuous-v0'] = 'MountainCarContinuous-v0'
    seed: int = 0

class MountainCarContinuous(gym.Env):
    def __init__(self, cfg: MountainCarContinuousConfig):
        self._cfg = cfg
        self._env = gym.make('MountainCarContinuous-v0')
        self._env.reset(seed=cfg.seed)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        return self._env.step(action)

    def close(self):
        self._env.close()

    def seed(self, seed: int):
        self._env.reset(seed=seed)

env_group.dispatcher(MountainCarContinuousConfig(), MountainCarContinuous)
