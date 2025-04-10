from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from coreenv.factory import EnvConfig, env_group


@dataclass
class StandStillMCConfig(EnvConfig):
    name: str = 'StandStillMC-v0'

class StandStillMC(gym.Env):
    def __init__(self, cfg: StandStillMCConfig):
        if cfg is None:
            cfg = StandStillMCConfig()
        self._env = gym.make("MountainCarContinuous-v0")


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        s, _, term, trunc, i = self._env.step(action)
        r = float(-1*action**2)
        return s, r, term, trunc, i


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        return self._env.reset(seed=seed, options=options)

    def close(self):
        self._env.close()

env_group.dispatcher(StandStillMCConfig(), StandStillMC)
