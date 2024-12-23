import random
import gymnasium as gym

from torch import Tensor
from typing import Tuple, Any

from corerl.models.base import BaseModel
from corerl.environment.config import EnvironmentConfig
from corerl.configs.config import MISSING, config


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
        self.curr_state: Tensor | None = None
        self.initial_states: list[Tensor] | None = None

    def set_model(self, model: BaseModel):
        self.model = model

    def set_initial_states(self, states: list[Tensor]):
        self.initial_states = states

    def step(self, action: Tensor) -> Tuple[Tensor, float, bool, bool, dict]:
        # TODO: figure out exo/endo?
        assert self.model is not None
        assert self.curr_state is not None
        assert self.rollout_step is not None
        next_obs, reward = self.model.predict(self.curr_state, action)
        # TODO: should rollout ends be termination or truncation?
        self.rollout_step += 1
        terminated = self.rollout_step == self.rollout_len
        return next_obs, reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> Tuple[Tensor, dict]:
        assert self.initial_states is not None
        self.rollout_step = 0
        self.curr_state = random.choice(self.initial_states)
        assert self.curr_state is not None
        return self.curr_state, {}
