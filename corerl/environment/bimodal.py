from typing import Any
import gymnasium as gym
import math
import numpy as np


class Bimodal(gym.Env):
    """
    The bimodal bandit environment from https://arxiv.org/abs/2107.08285, with
    state dimensions scaled by 2.
    """
    def __init__(self, seed, reward_variance=False):
        self._random = np.random.default_rng(seed)

        self._reward_variance = reward_variance
        self._state_dim = 1
        self._state_range = np.array([0.])
        self._state_min = np.array([0.])
        self._state_max = np.array([1.])
        self._state = np.array([1.])

        self.observation_space = gym.spaces.Box(
            self._state_min,
            self._state_max,
        )

        self._action_dim = 1
        self._action_range = np.array([4.])
        self._action_min = np.array([-2.])
        self._action_max = np.array([2.])

        self.action_space = gym.spaces.Box(self._action_min, self._action_max)

    def seed(self, seed):
        self._random = np.random.default_rng(seed)

    def step(self, action):
        action = action.item()
        self._state = self._state + action  # Terminal state

        # #####################
        # Calculate the reward
        # #####################
        maxima1 = -1.0
        maxima2 = 1.0
        stddev = 0.2

        # Reward function
        modal1 = 1. * math.exp(-0.5 * ((action - maxima1) / stddev)**2)
        modal2 = 1.5 * math.exp(-0.5 * ((action - maxima2) / stddev)**2)

        # Add some random noise to the reward with mean noise 0.
        # Use more variance at the lower mode so that it looks better,
        # but in expectation is worse.
        reward_var = 0
        if self._reward_variance:
            reward_var = self._random.normal(0.0, modal1 * 5.0)

        reward = modal1 + modal2 + reward_var

        return self._state, reward, True, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self._state = np.array([1.])
        return self._state, {}

    def close(self):
        pass
