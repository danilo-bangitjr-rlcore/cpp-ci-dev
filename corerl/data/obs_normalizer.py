from corerl.data.normalizer_utils import init_action_normalizer, init_reward_normalizer, init_obs_normalizer
from corerl.data.data import ObsTransition, OldObsTransition


class ObsTransitionNormalizer:
    def __init__(self, cfg, env):
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.obs_normalizer = init_obs_normalizer(cfg.obs_normalizer, env)

    def normalize(self, obs_transition: ObsTransition) -> ObsTransition:
        copy = obs_transition.copy()

        if isinstance(copy, OldObsTransition):
            copy.prev_action = self.action_normalizer(copy.prev_action)

        copy.obs = self.obs_normalizer(copy.obs)
        copy.action = self.action_normalizer(copy.action)
        copy.next_obs = self.obs_normalizer(copy.next_obs)
        copy.reward = self.reward_normalizer(copy.reward)
        return copy
