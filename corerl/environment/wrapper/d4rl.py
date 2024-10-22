import gym
import numpy as np
from gym.spaces import Box
from corerl.utils.gym import space_bounds

class D4RLWrapper:
    def __init__(self, name, seed):
        self.env = gym.make(name)
        self.env.unwrapped.seed(seed)
        obs_low, obs_high = space_bounds(self.env.observation_space)
        self.observation_space = Box(low=obs_low, high=obs_high)

        act_low, act_high = space_bounds(self.env.action_space)
        self.action_space = Box(low=act_low, high=act_high)

        # control timeout setting in the interface layer
        self.env._max_episode_steps = np.inf # type: ignore

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        ret = self.env.step(action)
        state, reward, done, info = ret
        return np.asarray(state), np.asarray(reward), np.asarray(done), False, info

    def reset(self, seed: int | None=None):
        if seed is not None:
            self.env.unwrapped.seed(seed)
        return self.env.reset(), {}

    def get_dataset(self) -> dict:
        data = self.env.get_dataset() # type: ignore
        data['rewards'] = data['rewards'].reshape((-1, 1))
        data['terminals'] = data['terminals'].reshape((-1, 1))
        return data
