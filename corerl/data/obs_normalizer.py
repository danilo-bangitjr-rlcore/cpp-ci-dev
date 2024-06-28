from corerl.data.normalizer_utils import init_action_normalizer, init_reward_normalizer, init_obs_normalizer
from corerl.data.data import ObsTransition


class ObsTransitionNormalizer:
    def __init__(self, cfg, env):
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.obs_normalizer = init_obs_normalizer(cfg.obs_normalizer, env)

    def normalize(self, obs_transition: ObsTransition) -> ObsTransition:
        obs_transition.prev_action = self.action_normalizer(obs_transition.prev_action)
        obs_transition.obs = self.obs_normalizer(obs_transition.obs)
        obs_transition.action = self.action_normalizer(obs_transition.action)
        obs_transition.next_obs = self.obs_normalizer(obs_transition.next_obs)
        obs_transition.reward = self.reward_normalizer(obs_transition.reward)
        return obs_transition
