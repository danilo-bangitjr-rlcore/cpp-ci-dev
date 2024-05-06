import gymnasium as gym
import numpy as np

# Revan: I don't remember what the purpose of this clas is
# To take the first action in the list.
# So that the action returned by the agent has consistent number of dimensions
# as in continuous control tasks. (1, action_dimension)
class DiscreteControlWrapper(gym.Env):
    def __init__(self, name, seed):
        self.env = gym.make(name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env._max_episode_steps = np.inf  # control timeout setting in agent
        self.env.reset(seed=seed)

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
    def __init__(self, name, seed):
        super(SparseDiscreteControlWrapper, self).__init__(name, seed)
        assert name in ['MountainCar-v0', 'Acrobot-v1'], "Only works for episodic tasks"

    def step(self, action):
        action_ = action[0]
        state, reward, done, truncate, env_info = self.env.step(action_)
        if reward == 0. or done:
            reward = 1
        else:
            reward = 0
        return state, reward, done, truncate, env_info