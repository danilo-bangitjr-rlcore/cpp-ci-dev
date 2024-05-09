import numpy as np
import gymnasium
from omegaconf import DictConfig

from corerl.interaction.base import BaseInteraction
from corerl.interaction.normalizer_utils import init_action_normalizer, init_reward_normalizer, init_obs_normalizer
from corerl.state_constructor.base import BaseStateConstructor


class NormalizerInteraction(BaseInteraction):
    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor,
            agent,
    ):
        super().__init__(cfg, env, state_constructor)
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, self.env, agent)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.obs_normalizer = init_obs_normalizer(cfg.obs_normalizer, env)

        self.gamma = cfg.gamma

    def step(self, state: np.ndarray, action: np.ndarray, decision_point=False) -> tuple[list[tuple], list[dict]]:
        # Revan: I'm not sure that this is the best place for decision_point ^
        denormalized_action = self.action_normalizer.denormalize(action)
        next_observation, raw_reward, terminated, env_truncate, env_info = self.env.step(denormalized_action)
        next_observation = self.obs_normalizer(next_observation)
        next_state = self.state_constructor(next_observation)
        reward = self.reward_normalizer(raw_reward)
        truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment

        if terminated or truncate:
            next_state, env_info = self.reset()  # if truncated or terminated, replace the next state and env_info with the return after resetting.

        gamma_exponent = 1
        return [(state, action, reward, next_state, terminated, truncate, decision_point, gamma_exponent)], [env_info]

    def reset(self) -> (np.ndarray, dict):
        observation, info = self.env.reset()
        self.state_constructor.reset()
        normalized_observation = self.obs_normalizer(observation)
        state = self.state_constructor(normalized_observation)
        return state, info

    def warmup_sc(self) -> None:
        """
        The state constructor warmup will be project specific.
        It will depend upon whether the environment is episodic/continuing.
        You might pass the recent history to the function and then loop self.state_constructor(obs)
        """
        pass
