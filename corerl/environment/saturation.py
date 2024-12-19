from typing import Any
import gymnasium as gym
import numpy as np


class Saturation(gym.Env):
    def __init__(self, **kwargs):
        self._random = np.random.default_rng()
        self.observation_dim = 1
        self._obs_min = np.array([0.])
        self._obs_max = np.array([1.])
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max)

        self.saturation = np.array([0.])
        self.saturation_sp = np.array([0.5])

        self.decay_period = 100

        self._action_dim = 1
        self._action_min = np.array([0])
        self._action_max = np.array([1])
        self.action_space = gym.spaces.Box(self._action_min, self._action_max)

        self.time_step = 0

        self.saturations = []
        self.decays = []
        self.actions = []

    def seed(self, seed):
        self._random = np.random.default_rng(seed)

    def step(self, action):
        self.time_step += 1
        decay = 0.15 * np.cos(self.time_step * np.pi * (2 / self.decay_period)) + 0.75
        self.saturation = self.saturation * decay + action
        self.saturation = np.clip(self.saturation, 0, 1)
        reward = -np.abs(self.saturation - self.saturation_sp).item()

        self.decays.append(decay)
        self.saturations.append(self.saturation)
        self.actions.append(action)

        return self.saturation, reward, False, False, {}

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.actions, label="actions")
        plt.plot(self.saturations, label="saturation")
        plt.plot(self.decays, label="decays")
        plt.legend()
        plt.show()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self.seed(seed)

        return self.saturation, {}

    def close(self):
        pass

gym.register('Saturation-v0', Saturation)
