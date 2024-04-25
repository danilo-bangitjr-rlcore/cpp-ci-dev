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


class SparseDiscreteControlWrapper(DiscreteControlWrapper):
    def __init__(self, name, timeout):
        super(SparseDiscreteControlWrapper, self).__init__(name, timeout)
        assert name in ['MountainCar-v0', 'Acrobot-v1'], "Only works for episodic tasks"

    def step(self, action):
        action_ = action[0]
        state, reward, done, truncate, env_info = self.env.step(action_)
        if reward == 0. or done:
            reward = 1
        else:
            reward = 0
        return state, reward, done, truncate, env_info