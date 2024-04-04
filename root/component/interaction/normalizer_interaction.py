import numpy as np
import gymnasium
from omegaconf import DictConfig
from abc import ABC, abstractmethod

from root.component.interaction.base_interaction import BaseInteraction
from root.component.interaction.normalizer import init_normalizer


class NormalizerInteraction(BaseInteraction):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env | BaseInteraction, state_constructor):
        super().__init__(cfg, env, state_constructor)
        self.env = env
        self.action_normalizer = init_normalizer(cfg.action_normalizer)
        self.reward_normalizer = init_normalizer(cfg.reward_normalizer)
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
