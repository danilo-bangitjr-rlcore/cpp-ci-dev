import gymnasium as gym
import numpy as np

class DelayedSaturation(gym.Env):
    def __init__(self, seed, trace_val):
        self._random = np.random.default_rng(seed)
        self.observation_dim = 1
        self._obs_min = np.array([0.])
        self._obs_max = np.array([1.])
        self._observation_space = gym.spaces.Box(self._obs_min, self._obs_max)

        self.saturation = np.array([0.])
        self.saturation_sp = np.array([0.5])
        self.effect_period = 500

        self._action_dim = 1
        self._action_min = np.array([0])
        self._action_max = np.array([1])
        self._action_space = gym.spaces.Box(self._action_min, self._action_max)

        self.time_step = 0
        self.decay = 0.75

        self.saturations = []
        self.effects = []
        self.actions = []
        self.raw_actions = []
        self.action_trace = np.array([0])
        self.trace_val = trace_val

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

        self.action_trace = self.trace_val*self.action_trace + (1-self.trace_val)*action
        self.saturation = self.saturation * self.decay
        effect = 0.15 * np.cos(self.time_step * np.pi * (2 / self.effect_period)) + 0.75
        self.saturation = self.action_trace*effect
        self.saturation = np.clip(self.saturation, 0, 1)
        reward = -np.abs(self.saturation - self.saturation_sp).item()

        self.effects.append(effect)
        self.saturations.append(self.saturation)
        self.raw_actions.append(action)
        self.actions.append(self.action_trace)

        return self.saturation, reward, False, False, {}

    def plot(self, save_path):
        import matplotlib.pyplot as plt
        plt.plot(self.raw_actions, label="raw actions")
        plt.plot(self.actions, label="actions")
        plt.plot(self.saturations, label="saturation")
        plt.plot(self.effects, label="effects")
        plt.legend()
        plt.savefig(save_path/'env.png', bbox_inches='tight')
        # plt.show()

    def reset(self):
        return self.saturation, {}

    def close(self):
        pass

# class DelayedSaturation(gym.Env):
#     def __init__(self, seed, trace_val):
#         self._random = np.random.default_rng(seed)
#         self.observation_dim = 1
#         self._obs_min = np.array([0.])
#         self._obs_max = np.array([1.])
#         self._observation_space = gym.spaces.Box(self._obs_min, self._obs_max)
#
#         self.saturation = np.array([0.])
#         self.saturation_sp = np.array([0.5])
#         self.decay_period = 500
#
#         self._action_dim = 1
#         self._action_min = np.array([0])
#         self._action_max = np.array([1])
#         self._action_space = gym.spaces.Box(self._action_min, self._action_max)
#
#         self.time_step = 0
#
#         self.saturations = []
#         self.decays = []
#         self.actions = []
#         self.raw_actions = []
#         self.action_trace = np.array([0])
#         self.trace_val = trace_val
#
#     def seed(self, seed):
#         self._random = np.random.default_rng(seed)
#
#     @property
#     def action_space(self):
#         return self._action_space
#
#     @property
#     def observation_space(self):
#         return self._observation_space
#
#     def step(self, action):
#         self.time_step += 1
#         decay = 0.15 * np.cos(self.time_step * np.pi * (2 / self.decay_period)) + 0.75
#
#         self.action_trace = self.trace_val*self.action_trace + (1-self.trace_val)*action
#
#         self.saturation = self.saturation * decay + self.action_trace/2
#         self.saturation = np.clip(self.saturation, 0, 1)
#         reward = -np.abs(self.saturation - self.saturation_sp).item()
#
#         self.decays.append(decay)
#         self.saturations.append(self.saturation)
#         self.raw_actions.append(action)
#         self.actions.append(self.action_trace)
#
#         return self.saturation, reward, False, False, {}
#
#     def plot(self, save_path):
#         import matplotlib.pyplot as plt
#         plt.plot(self.raw_actions, label="raw actions")
#         plt.plot(self.actions, label="actions")
#         plt.plot(self.saturations, label="saturation")
#         plt.plot(self.decays, label="decays")
#         plt.legend()
#         plt.savefig(save_path/'env.png', bbox_inches='tight')
#         # plt.show()
#
#     def reset(self):
#         return self.saturation, {}
#
#     def close(self):
#         pass