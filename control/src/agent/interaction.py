from src.component.normalizer import init_normalizer
import types


class InteractionLayer:
    def __init__(self, cfg):
        self.env = cfg.train_env
        self.eval_env = cfg.eval_env
        self.observation, info = self.env.reset(seed=cfg.seed)
        self.eval_env.reset(seed=cfg.seed)

        self.state_normalizer = init_normalizer(cfg.state_normalizer, self.env.observation_space)
        self.reward_normalizer = init_normalizer(cfg.reward_normalizer, None)
        self.action_normalizer = init_normalizer(cfg.action_normalizer,
                                                 obj=type('obj', (object,), {'scaler': cfg.action_scale,  'bias': cfg.action_bias}))

    def env_reset(self):
        raw_observation, info = self.env.reset()
        return self.state_normalizer(raw_observation), info

    def env_step(self, action):
        raw_action = self.action_normalizer.denormalizer(action)
        raw_observation, raw_reward, done, truncate, env_info = self.env.step(raw_action)
        return self.state_normalizer(raw_observation), self.reward_normalizer(raw_reward), done, truncate, env_info

    def eval_reset(self):
        raw_observation, info = self.eval_env.reset()
        return self.state_normalizer(raw_observation), info

    def eval_step(self, action):
        raw_action = self.action_normalizer.denormalizer(action)
        raw_observation, raw_reward, done, truncate, env_info = self.eval_env.step(raw_action)
        return self.state_normalizer(raw_observation), self.reward_normalizer(raw_reward), done, truncate, env_info

    def get_action_samples(self, n=50):
        action_cover_space, heatmap_shape = self.env.get_action_samples(n=n)
        return self.action_normalizer(action_cover_space)