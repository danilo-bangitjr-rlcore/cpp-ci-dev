import gym
from gym.spaces import Box
import numpy as np

class D4RLWrapper:
    def __init__(self, name, seed):
        self.env = gym.make(name)
        self.env.unwrapped.seed(seed)
        self.observation_space = Box(low=self.env.observation_space.low, high=self.env.observation_space.high)
        self.action_space = Box(low=self.env.action_space.low, high=self.env.action_space.high)
        self.env._max_episode_steps = np.inf  # control timeout setting in agent

    def step(self, action):
        ret = self.env.step(action)
        state, reward, done, info = ret
        return np.asarray(state), np.asarray(reward), np.asarray(done), False, info

    def reset(self, seed=None):
        if seed is not None:
            self.env.unwrapped.seed(seed)
        return self.env.reset(), {}

    def get_dataset(self):
        data = self.env.get_dataset()
        data['rewards'] = data['rewards'].reshape((-1, 1))
        data['terminals'] = data['terminals'].reshape((-1, 1))
        return data