import gymnasium as gym
import numpy as np

# Revan: I don't remember what the purpose of this clas is
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
