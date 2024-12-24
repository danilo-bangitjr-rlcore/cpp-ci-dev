import random
from typing import Any, Tuple

import gymnasium as gym
import torch

from corerl.configs.config import MISSING, config
from corerl.environment.config import EnvironmentConfig
from corerl.models.base import BaseModel


@config()
class ModelEnvConfig(EnvironmentConfig):
    type: str = 'custom'
    name: str = 'model_env'
    rollout_len: int = MISSING


class ModelEnv(gym.Env):
    def __init__(self, cfg: ModelEnvConfig):
        self.rollout_len: int = cfg.rollout_len
        self.model: BaseModel | None = None
        self.rollout_step: int | None = None
        self.curr_state: torch.Tensor | None = None
        self.initial_states: list[torch.Tensor] | None = None
        # list of lists of exogenous observations
        self.exo_obs_seqs: list[list[torch.Tensor]] | None = None
        # list of exogenous observations for current rollout
        self.curr_exo_seq: list[torch.Tensor] | None = None

    def set_model(self, model: BaseModel):
        self.model = model

    def set_initial_states(self,
                           states: list[torch.Tensor],
                           exo_obs_seqs: list[list[torch.Tensor]] | None = None,
                           ):
        """
        Sets initial starting points for rollouts. If using an endogenous model,
        specify exo_obs_seqs as the sequence of exogenous observations for the rollouts.
        There should be the same number of states and exo_obs_seqs.
        The first element in each list in exo_obs_seqs comes one timestep AFTER the corresponding state.
        """
        self.initial_states = states
        if exo_obs_seqs is not None:
            assert len(states) == len(exo_obs_seqs)
            for exo_obs_seq in exo_obs_seqs:
                assert len(exo_obs_seq) == self.rollout_len, ("Please give sequences of exogenous observations "
                                                              "at least as long as rollout_len")
        self.exo_obs_seqs = exo_obs_seqs

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        assert self.model is not None
        assert self.curr_state is not None
        assert self.rollout_step is not None
        assert self.rollout_step <= self.rollout_len, ("self.rollout_step <= self.rollout_len, "
                                                       "did you forget to call reset()?")
        next_obs, reward = self.model.predict(self.curr_state, action)

        if self.curr_exo_seq is not None:
            exo_obs = self.curr_exo_seq[self.rollout_step]
            next_obs = torch.concat([next_obs, exo_obs])

        self.rollout_step += 1
        terminated = self.rollout_step == self.rollout_len  # TODO: should rollout ends be termination or truncation?
        return next_obs, reward, terminated, False, {}

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> Tuple[torch.Tensor, dict]:

        assert self.initial_states is not None
        assert self.model is not None
        self.rollout_step = 0
        rollout_idx = random.choice(range(len(self.initial_states)))
        self.curr_state = self.initial_states[rollout_idx]
        if self.exo_obs_seqs is not None:
            assert self.model.endogenous, ("You included exogenous observations but are not using an endogenous model,"
                                           "exogenous observations would be duplicated.")
            self.curr_exo_seq = self.exo_obs_seqs[rollout_idx]

        assert self.curr_state is not None
        return self.curr_state, {}
