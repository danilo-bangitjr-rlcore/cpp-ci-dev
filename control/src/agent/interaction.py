from gymnasium.spaces.utils import flatdim
from src.component.normalizer import init_normalizer
from src.component.state_constructor import init_state_constructor
import types


class InteractionLayer:
    def __init__(self, cfg):
        self.env = cfg.train_env
        self.eval_env = cfg.eval_env
        self.observation, info = self.env.reset(seed=cfg.seed)
        self.eval_env.reset(seed=cfg.seed)

        #self.state_normalizer = init_normalizer(cfg.state_normalizer, self.env.observation_space)
        self.reward_normalizer = init_normalizer(cfg.reward_normalizer, None)
        self.action_normalizer = init_normalizer(cfg.action_normalizer,
                                                 type('obj', (object,), {'scaler': cfg.action_scale,  'bias': cfg.action_bias}))
        
        self.state_constructor = init_state_constructor(cfg.state_constructor, cfg)
        self.state_dim = self.state_constructor.get_state_dim(flatdim(self.env.observation_space))   
        print('State_dim: {}'.format(self.state_dim))
        self.action_dim = self.action_normalizer.get_new_dim(flatdim(self.env.action_space))

    def env_reset(self):
        raw_observation, info = self.env.reset()
        return self.state_constructor(raw_observation), info

    def env_step(self, action):
        raw_action = self.action_normalizer.denormalize(action)[0]
        raw_observation, raw_reward, done, truncate, env_info = self.env.step(raw_action)
        return self.state_constructor(raw_observation), self.reward_normalizer(raw_reward), done, truncate, env_info

    def take_action(self, action):
        raw_action = self.action_normalizer.denormalize(action)[0]
        self.env.take_action(raw_action)

    def get_observation(self, action):
        raw_action = self.action_normalizer.denormalize(action)[0]
        raw_observation, raw_reward, done, truncate, env_info = self.env.get_observation(raw_action)  
        state = self.state_constructor(raw_observation)
        return state, self.reward_normalizer(raw_reward), done, truncate, env_info

    def evalenv_reset(self):
        raw_observation, info = self.eval_env.reset()
        return self.state_constructor(raw_observation), info

    def evalenv_step(self, action):
        raw_action = self.action_normalizer.denormalize(action)[0]
        raw_observation, raw_reward, done, truncate, env_info = self.eval_env.step(raw_action)
        return self.state_constructor(raw_observation), self.reward_normalizer(raw_reward), done, truncate, env_info

    def get_action_samples(self, n=50):
        action_cover_space, heatmap_shape = self.env.get_action_samples(n=n)
        return self.action_normalizer(action_cover_space), heatmap_shape


    
