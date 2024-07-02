from copy import deepcopy
from corerl.data.normalizer_utils import init_action_normalizer, init_reward_normalizer, init_obs_normalizer


class TransitionNormalizer:
    def __init__(self, cfg, env):
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.obs_normalizer = init_obs_normalizer(cfg.obs_normalizer, env)

    def denormalize(self, transition):
        transition_copy = deepcopy(transition)
        
        transition_copy.obs = self.obs_normalizer.denormalize(transition_copy.obs)
        transition_copy.action = self.action_normalizer.denormalize(transition_copy.action)
        transition_copy.next_obs = self.obs_normalizer.denormalize(transition_copy.next_obs)
        transition_copy.reward = self.reward_normalizer.denormalize(transition_copy.reward)
        
        return transition_copy
