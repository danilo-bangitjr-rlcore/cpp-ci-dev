import gymnasium as gym
import gym_electric_motor as gem
from gymnasium.wrappers import FlattenObservation
import numpy as np

class DiscreteControlWrapper(gym.Env):
    def __init__(self, name):
        self.env = gym.make(name)
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
