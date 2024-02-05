from gymnasium.spaces.utils import flatdim
from src.component.normalizer import init_normalizer
import types


class InteractionLayer:
    def __init__(self, cfg):
        self.env = cfg.train_env
        self.eval_env = cfg.eval_env
        self.observation, info = self.env.reset(seed=cfg.seed)
        self.eval_env.reset(seed=cfg.seed)

        self.state_normalizer = init_normalizer(cfg.state_normalizer, self.env.observation_space)
        # reward_normalizer = cfg.reward_normalizer.split("/")
        # self.reward_normalizer = init_normalizer(reward_normalizer[0], [float(i) for i in reward_normalizer[1:]])
        self.reward_normalizer = init_normalizer(cfg.reward_normalizer, None)
        self.action_normalizer = init_normalizer(cfg.action_normalizer, [cfg.action_scale, cfg.action_bias])

        self.state_dim = self.state_normalizer.get_new_dim(flatdim(self.env.observation_space))
        self.action_dim = self.action_normalizer.get_new_dim(flatdim(self.env.action_space))

    def env_reset(self):
        raw_observation, info = self.env.reset()
        return self.state_normalizer(raw_observation), info

    def env_step(self, action):
        raw_action = self.action_normalizer.denormalize(action)[0]
        raw_observation, raw_reward, done, truncate, env_info = self.env.step(raw_action)
        return self.state_normalizer(raw_observation), self.reward_normalizer(raw_reward), done, truncate, env_info

    def evalenv_reset(self):
        raw_observation, info = self.eval_env.reset()
        return self.state_normalizer(raw_observation), info

    def evalenv_step(self, action):
        raw_action = self.action_normalizer.denormalize(action)[0]
        raw_observation, raw_reward, done, truncate, env_info = self.eval_env.step(raw_action)
        return self.state_normalizer(raw_observation), self.reward_normalizer(raw_reward), done, truncate, env_info

    def get_action_samples(self, n=50):
        action_cover_space, heatmap_shape = self.env.get_action_samples(n=n)
        return self.action_normalizer(action_cover_space), heatmap_shape