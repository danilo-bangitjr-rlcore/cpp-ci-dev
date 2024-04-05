import numpy as np
import gymnasium
from omegaconf import DictConfig

from root.interaction.base import BaseInteraction
from root.interaction.normalizer_utils import init_action_normalizer, init_reward_normalizer


class NormalizerInteraction(BaseInteraction):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env, state_constructor):
        super().__init__(cfg, env, state_constructor)
        self.env = env
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, self.env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.state_constructor = state_constructor

    def step(self, action: np.ndarray) -> tuple:
        denormalized_action = self.action_normalizer.denormalize(action)
        next_observation, reward, terminated, truncate, env_info = self.env.step(denormalized_action)
        next_state = self.state_constructor(next_observation)
        return next_state, reward, terminated, truncate, env_info

    def reset(self) -> (np.ndarray, dict):
        observation, info = self.env.reset()
        state = self.state_constructor(observation)
        return state, info
