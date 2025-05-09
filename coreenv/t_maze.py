import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from coreenv.factory import EnvConfig, env_group


@dataclass
class TMazeConfig(EnvConfig):
    name: str = 'TMaze-v0'
    corridor_length: float = 1.0
    end_zone_size: float = 0.1


OBS_DIM = 3 # the initial signal, the pos, and whether we are in the endzone
DELTA = 0.1

class TMaze(gym.Env):
    def __init__(self, cfg: TMazeConfig ):
        if cfg is None:
            cfg = TMazeConfig()

        self._cfg = cfg
        self._random = np.random.default_rng(cfg.seed)
        self._obs_min = np.zeros(OBS_DIM)
        self._obs_max = np.ones(OBS_DIM)
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        self._action_min = -np.ones(1)
        self._action_max = np.ones(1)
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)

        self._corridor_length = cfg.corridor_length
        self._end_zone = cfg.end_zone_size
        self._end_zone_start = self._corridor_length * (1-self._end_zone)

    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)


    def _in_endzone(self):
        return self.pos > self._end_zone_start


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        float_action = float(action[0])
        if self._in_endzone():
            action_is_left = float_action < 0
            reward = 0 if action_is_left == self.signal_is_left else -1
            state, _ = self.reset()
            return state, reward, False, False, {}

        else:
            self.pos += DELTA*float_action
            self.pos = np.maximum(0, self.pos)

            if float_action > 0:
                reward = 0
            else:
                reward = 0.1*float_action

            return np.array([0, self.pos, self._in_endzone()]), reward, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:

        signal = -1 if random.random() < 0.5 else 1
        self.signal_is_left = signal < 0
        self.pos = 0.0
        state = np.array([signal, self.pos, 0.0])
        return state, {}

    def close(self):
        pass

env_group.dispatcher(TMazeConfig(), TMaze)
