import gymnasium as gym
import numpy as np


class Saturation(gym.Env):
    def __init__(self, seed):
        self._random = np.random.default_rng(seed)
        self.observation_dim = 1
        self._obs_min = np.array([0.])
        self._obs_max = np.array([1.])
        self._observation_space = gym.spaces.Box(self._obs_min, self._obs_max)

        self.saturation = np.array([0.])
        self.saturation_sp = np.array([0.5])
        # self.sensitivity =  np.array([1.])
        #
        # self.sensitivity_min = np.array([0.])
        # self.sensitivity_max = np.array([1.])

        # self.decay = np.array([0.95])
        self.decay_period = 200

        self._action_dim = 1
        self._action_min = np.array([0])
        self._action_max = np.array([1])
        self._action_space = gym.spaces.Box(self._action_min, self._action_max)

        self.time_step = 0

        self.saturations = []
        self.decays = []
        self.actions = []

    def seed(self, seed):
        self._random = np.random.default_rng(seed)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

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

    def reset(self):
        return self.saturation, {}

    def close(self):
        pass
