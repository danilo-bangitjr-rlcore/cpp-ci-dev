from dataclasses import dataclass
from datetime import UTC, datetime
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
import pandas as pd

from corerl.configs.config import config
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, GymEnvConfig
from corerl.utils.gym import space_bounds, space_shape


@config()
class SimAsyncEnvConfig(GymEnvConfig):
    name: str = "sim_async_env"

@dataclass
class StepData:
    observation: np.ndarray
    reward: SupportsFloat
    action: np.ndarray
    truncated: bool
    terminated: bool


class SimAsyncEnv(AsyncEnv):
    def __init__(self, cfg: SimAsyncEnvConfig, tags: list[TagConfig]):
        self._env = gym.make(cfg.gym_name, *cfg.args, **cfg.kwargs)
        self._cfg = cfg

        shape = self._env.observation_space.shape
        assert shape is not None
        assert len(shape) == 1, "Cannot handle environments with non-vector observations"

        self._action_bounds = space_bounds(self._env.action_space)
        self._action_shape = space_shape(self._env.action_space)

        self.tags = tags

        self._action_tag_names = [tag.name for tag in tags if tag.is_action]
        self._observation_tag_names = [tag.name for tag in tags if not tag.is_action and not tag.is_meta]
        self._meta_tag_names = [tag.name for tag in tags if tag.is_meta]

        self.clock = datetime(1984, 1, 1, tzinfo=UTC)
        self._clock_inc = cfg.obs_period

        self._action: np.ndarray | None = None
        self._last_step: StepData | None = None

    # ------------------
    # -- AsyncEnv API --
    # ------------------
    def emit_action(self, action: np.ndarray) -> None:
        lo, hi = self._action_bounds
        scale = hi - lo
        bias = lo
        self._action = scale * action + bias

    def get_latest_obs(self) -> pd.DataFrame:
        self.clock += self._clock_inc

        if self._action is None or self._last_step is None or (self._last_step.terminated or self._last_step.truncated):
            step = self._reset()
        else:
            step = self._step(self._action)

        return self._obs_as_df(step)

    # -------------------------
    # -- Gym API Translation --
    # -------------------------
    def _reset(self):
        observation, _info = self._env.reset()

        self._last_step = StepData(
            observation=observation,
            reward=np.nan,
            action=np.full(self._action_shape, np.nan),
            truncated=False,
            terminated=False,
        )

        return self._last_step

    def _step(self, action: np.ndarray):
        observation, reward, terminated, truncated, _info = self._env.step(action)
        self._last_step = StepData(
            observation=observation,
            reward=reward,
            action=action,
            truncated=truncated,
            terminated=terminated,
        )
        return self._last_step

    def _obs_as_df(self, step: StepData):
        obs_data = {tag: val for tag, val in zip(self._observation_tag_names, step.observation, strict=True)}

        action_data = {tag: val for tag, val in zip(self._action_tag_names, step.action, strict=True)}

        meta_data = {
            "reward": step.reward,
            "truncated": step.truncated,
            "terminated": step.terminated,
        }

        idx = pd.DatetimeIndex([self.clock])
        df = pd.DataFrame(obs_data | action_data | meta_data, index=idx)
        return df
