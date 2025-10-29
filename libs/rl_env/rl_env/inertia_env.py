from typing import Any, Literal

import gymnasium as gym
import numpy as np
from lib_config.config import config

from rl_env.group_util import EnvConfig, env_group


@config(frozen=True)
class InertiaEnvConfig(EnvConfig):
    name: Literal["InertiaEnv-v0"] = "InertiaEnv-v0"
    action_dim: int = 1
    target: float = 0.7


class InertiaEnv(gym.Env):
    """
    Environment where the agent's goal is to learn to select the 'target' action.
    The observation is the action taken on the previous time step.
    The agent receives a reward based on how close its action is to the target and
    is also given an action change penalty.
    In configuration, the agent is given a nominal setpoint bias of 0.15. The agent must sufficiently
    explore to overcome the preference for increased determinism along the operating range boundary in favour
    of finding larger rewards near the target.
    """

    def __init__(self, cfg: InertiaEnvConfig):
        self._random = np.random.default_rng(cfg.seed)
        self.action_dim = cfg.action_dim
        self.target = cfg.target

        # Observation space: between 0 and 1 for each dimension
        self._obs_min = np.zeros(self.action_dim)
        self._obs_max = np.ones(self.action_dim)
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        # Action space: between 0 and 1 for each dimension
        self._action_min = np.zeros(self.action_dim)
        self._action_max = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)

        # Initial state: randomly initialized between 0 and 1 for all dimensions
        self.state = self._random.uniform(0.0, 1.0, size=self.action_dim)

    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)

    def step(self, action: np.ndarray):
        # Action change penalty component of the reward
        reward = -np.sum(np.abs(self.state - action))

        # Proximity to target action reward component
        reward -= 0.2 * np.abs(self.state - self.target)

        # State becomes the current action
        self.state = action.copy()

        return self.state.copy(), reward, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self.seed(seed)

        # Reset state to random values between 0 and 1 for all dimensions
        self.state = self._random.uniform(0.0, 1.0, size=self.action_dim)
        return self.state.copy(), {}

    def close(self):
        pass


env_group.dispatcher(InertiaEnvConfig(), InertiaEnv)
