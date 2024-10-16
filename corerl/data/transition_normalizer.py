import logging
import gymnasium as gym
from copy import deepcopy
from omegaconf import DictConfig
from corerl.data.data import Transition
from corerl.data.normalizer_utils import InvertibleNormalizer, init_action_normalizer, init_reward_normalizer, init_obs_normalizer

log = logging.getLogger(__name__)

class TransitionNormalizer:
    def __init__(self, cfg: DictConfig, env: gym.Env):
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.obs_normalizer = init_obs_normalizer(cfg.obs_normalizer, env)

    def denormalize(self, transition: Transition):
        transition_copy = deepcopy(transition)

        transition_copy.obs = self.obs_normalizer.denormalize(transition_copy.obs)
        transition_copy.action = self.action_normalizer.denormalize(transition_copy.action)
        transition_copy.next_obs = self.obs_normalizer.denormalize(transition_copy.next_obs)

        if isinstance(self.reward_normalizer, InvertibleNormalizer):
            transition_copy.reward = self.reward_normalizer.denormalize(transition_copy.reward)
        else:
            logging.warning('Tried to invert a reward that was normalized with a non-invertible normalizer')

        return transition_copy
