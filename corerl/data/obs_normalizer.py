from corerl.data.normalizer_utils import init_action_normalizer, init_reward_normalizer, init_obs_normalizer
from corerl.data.data import ObsTransition, OldObsTransition

from copy import copy

class ObsTransitionNormalizer:
    def __init__(self, cfg, env):
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.obs_normalizer = init_obs_normalizer(cfg.obs_normalizer, env)

    def normalize(self, obs_transition: ObsTransition) -> ObsTransition:
        obs_transition_copy = copy(obs_transition)

        if isinstance(obs_transition_copy, OldObsTransition):
            obs_transition_copy.prev_action = self.action_normalizer(obs_transition_copy.prev_action)

        obs_transition_copy.obs = self.obs_normalizer(obs_transition_copy.obs)
        obs_transition_copy.action = self.action_normalizer(obs_transition_copy.action)
        obs_transition_copy.next_obs = self.obs_normalizer(obs_transition_copy.next_obs)
        obs_transition_copy.reward = self.reward_normalizer(obs_transition_copy.reward)
        return obs_transition_copy
