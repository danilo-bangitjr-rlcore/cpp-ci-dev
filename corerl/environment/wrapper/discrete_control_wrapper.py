from typing import Any
import gymnasium as gym
import numpy as np

class DiscreteControlWrapper(gym.Env):
    def __init__(self, name, seed):
        self.env = gym.make(name, render_mode="human")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # control timeout setting in agent
        self.env._max_episode_steps = np.inf # type: ignore
        self.env.reset(seed=seed)

    def step(self, action):
        action_ = action[0]
        return self.env.step(action_)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        return self.env.reset(seed=seed)

    def get_action_samples(self):
        assert isinstance(self.action_space, gym.spaces.Discrete)
        samples = np.arange(self.action_space.n).reshape(-1, 1)
        shape = samples.shape
        return samples, shape

    def render(self):
        return self.env.render()


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
