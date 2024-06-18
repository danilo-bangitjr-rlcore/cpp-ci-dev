import numpy as np
import gymnasium
from omegaconf import DictConfig
from gymnasium.spaces.utils import flatdim

from corerl.interaction.base import BaseInteraction
from corerl.interaction.normalizer_utils import init_action_normalizer, init_reward_normalizer, init_obs_normalizer
from corerl.state_constructor.base import BaseStateConstructor
from corerl.data import Transition


class NormalizerInteraction(BaseInteraction):
    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor
    ):

        # can't I just get the action dim from the env?


        super().__init__(cfg, env, state_constructor)
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, self.env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.obs_normalizer = init_obs_normalizer(cfg.obs_normalizer, env)
        self.gamma = cfg.gamma
        self.last_state = None
        self.last_obs = None
        self.action_dim = flatdim(env.action_space)

    def step(self, action: np.ndarray) -> tuple[list[Transition], list[dict]]:
        # Revan: I'm not sure that this is the best place for decision_point ^
        # also adding next_decision_point, which is whether the next state is a decision point.
        denormalized_action = self.action_normalizer.denormalize(action)

        next_obs, raw_reward, terminated, env_truncate, env_info = self.env.step(denormalized_action)
        normalized_next_obs = self.obs_normalizer(next_obs)
        next_state = self.state_constructor(normalized_next_obs, action)
        reward = self.reward_normalizer(raw_reward)
        truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment
        gamma_exponent = 1

        transition = Transition(
            self.last_obs,
            self.last_state,
            action,
            normalized_next_obs,
            next_state,
            reward,
            normalized_next_obs,  # the obs for bootstrapping is the same as the next obs here
            next_state,  # the state for bootstrapping is the same as the next state here
            terminated,
            truncate,
            True,  # always a decision point
            True,  # always a decision point
            gamma_exponent)

        self.last_state = next_state
        self.last_obs = next_obs

        return [transition], [env_info]

    def reset(self) -> (np.ndarray, dict):
        observation, info = self.env.reset()
        self.state_constructor.reset()

        normalized_observation = self.obs_normalizer(observation)
        dummy_action = np.zeros(self.action_dim)
        # assume the initial state is always a decision point
        state = self.state_constructor(normalized_observation, dummy_action, initial_state=True, decision_point=True)

        self.last_obs = normalized_observation
        self.last_state = state

        return state, info

    def warmup_sc(self) -> None:
        """
        The state constructor warmup will be project specific.
        It will depend upon whether the environment is episodic/continuing.
        You might pass the recent history to the function and then loop self.state_constructor(obs)
        """
        pass
