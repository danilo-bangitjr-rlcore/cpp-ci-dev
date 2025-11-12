from typing import Any, Literal

import gymnasium as gym
import numpy as np
from lib_config.config import config

from rl_env.group_util import EnvConfig, env_group


@config(frozen=True)
class LostInTranslationConfig(EnvConfig):
    name: Literal["LostInTranslation-v0"] = "LostInTranslation-v0"
    delta: float = 0.0


class LostInTranslation(gym.Env):
    """
    Environment where state evolves randomly and agent must match state with action.
    Used to test critic translation.
    """

    def __init__(self, cfg: LostInTranslationConfig):
        self._random = np.random.default_rng(cfg.seed)
        self.delta = cfg.delta

        self._obs_min = np.array([0.0])
        self._obs_max = np.array([1.0])
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        self._action_min = np.array([0.0])
        self._action_max = np.array([1.0])
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)

        self.state = np.array([0.5])

    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)

    def step(self, action: np.ndarray):
        # next state is sampled randomly
        offset = self._random.uniform(-self.delta, self.delta)
        self.state = np.clip(self.state + offset, 0.0, 1.0)

        reward = -10.0 - np.abs(self.state - action).item()

        return self.state.copy(), reward, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self.seed(seed)

        self.state = np.array([0.5])
        return self.state.copy(), {}

    def close(self):
        pass


env_group.dispatcher(LostInTranslationConfig(), LostInTranslation)
