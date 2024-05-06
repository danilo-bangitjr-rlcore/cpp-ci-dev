import numpy as np
import gymnasium
from omegaconf import DictConfig

from corerl.interaction.base import BaseInteraction
from corerl.interaction.normalizer_utils import init_action_normalizer, init_reward_normalizer
from corerl.state_constructor.base import BaseStateConstructor


class NormalizerInteraction(BaseInteraction):
    def __init__(self, cfg: DictConfig, env: gymnasium.Env, state_constructor: BaseStateConstructor):
        super().__init__(cfg, env, state_constructor)
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, self.env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[list[tuple], list[dict]]:
        denormalized_action = self.action_normalizer.denormalize(action)
        next_observation, raw_reward, terminated, env_truncate, env_info = self.env.step(denormalized_action)
        reward = self.reward_normalizer(raw_reward)
        next_state = self.state_constructor(next_observation)
        reward = self.reward_normalizer(reward)
        truncate = self.env_counter() # use the interaction counter to decide reset. Remove reset in environment
        if terminated or truncate:
            next_state, env_info = self.reset() # if truncated or terminated, replace the next state and env_info with the return after resetting.
        return [(state, action, reward, next_state, terminated, truncate)], [env_info]

    def reset(self) -> (np.ndarray, dict):
        observation, info = self.env.reset()
        self.state_constructor.reset()
        state = self.state_constructor(observation)
        return state, info

    def warmup_sc(self) -> None:
        """
        The state constructor warmup will be project specific.
        It will depend upon whether the environment is episodic/continuing.
        You might pass the recent history to the function and then loop self.state_constructor(obs)
        """
        pass