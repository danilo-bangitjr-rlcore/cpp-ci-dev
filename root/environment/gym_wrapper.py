import gymnasium as gym
import gym_electric_motor as gem
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces import Box
import numpy as np

class DiscreteControlWrapper(gym.Env):
    def __init__(self, name, timeout):
        self.env = gym.make(name, max_episode_steps=timeout)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def step(self, action):
        action_ = action[0]
        return self.env.step(action_)
        
    def reset(self, seed=0):
        return self.env.reset(seed=seed)

    def get_action_samples(self, n=None):
        samples = np.arange(self.action_space.n).reshape(-1, 1)
        shape = samples.shape
        return samples, shape


class D4RLWrapper:

    def __init__(self, name, seed):
        import d4rl
        import gym
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