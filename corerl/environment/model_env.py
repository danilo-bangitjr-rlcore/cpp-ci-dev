import random

from torch import Tensor

from corerl.models.base import BaseModel
from corerl.environment.config import EnvironmentConfig
from corerl.configs.config import MISSING, config
from corerl.data_pipeline.datatypes import Transition


@config()
class ModelEnvConfig(EnvironmentConfig):
    type: str = 'custom'
    name: str = 'model_env'
    rollout_len: int = MISSING


class ModelEnv:
    def __init__(self, cfg: ModelEnvConfig):
        self.episode_len: int = cfg.rollout_len
        self.model: BaseModel | None = None
        self.step: int | None = None
        self.curr_state: None = None
        self.initial_states: list[Tensor] | None = None

    def set_model(self, model: BaseModel):
        self.model = model

    def set_initial_states(self, transitions: Transition):
        self.initial_states = [t.prior.state for t in transitions]

    def step(self, action):
        # TODO: figure out exo/endo?
        assert self.model is not None
        self.step += 1
        next_obs, reward = self.model.predict(self.curr_state, action)
        # TODO: should rollout ends be termination or truncation?
        terminated = self.step == self.episode_len
        return next_obs, reward, terminated, False, {}

    def reset(self):
        return random.choice(self.initial_states), {}
