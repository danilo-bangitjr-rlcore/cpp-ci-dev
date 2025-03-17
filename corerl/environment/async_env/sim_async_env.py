from dataclasses import dataclass
from datetime import UTC, datetime
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
import pandas as pd

from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, SimAsyncEnvConfig
from corerl.utils.gym import space_bounds, space_shape


@dataclass
class StepData:
    observation: np.ndarray
    reward: SupportsFloat
    action: np.ndarray
    truncated: bool
    terminated: bool


class SimAsyncEnv(AsyncEnv):
    """AsyncEnv which directly runs and steps through a Farama Gymnasium environment.
    """
    def __init__(self, cfg: SimAsyncEnvConfig, tags: list[TagConfig]):
        kwargs = dict(cfg.kwargs)
        if cfg.env_config is not None:
            if 'seed' in cfg.env_config and cfg.env_config['seed'] is not None:
                # manually to sync with experiment seed
                # TODO: remove this once we have a better way to handle this
                cfg.env_config['seed'] = cfg.seed
            kwargs['cfg'] = cfg.env_config
            if 'env_config' in kwargs:
                del kwargs['env_config']

        self._env = gym.make(cfg.gym_name, *cfg.args, **kwargs)
        self._cfg = cfg

        shape = self._env.observation_space.shape
        assert shape is not None
        assert len(shape) == 1, "Cannot handle environments with non-vector observations"

        self._action_bounds = space_bounds(self._env.action_space)
        self._action_shape = space_shape(self._env.action_space)

        self.tags = tags

        self._action_tag_names = [tag.name for tag in tags if tag.action_constructor is not None]
        self._meta_tag_names = [tag.name for tag in tags if tag.is_meta]
        self._observation_tag_names = [
            tag.name for tag in tags
            if not tag.is_meta and tag.action_constructor is None
        ]

        self._all_tag_names = set(self._action_tag_names + self._meta_tag_names + self._observation_tag_names)

        self.clock = datetime(1984, 1, 1, tzinfo=UTC)
        self._clock_inc = cfg.obs_period

        self._action: np.ndarray | None = None
        self._last_step: StepData | None = None

    # ------------------
    # -- AsyncEnv API --
    # ------------------
    def emit_action(self, action: pd.DataFrame) -> None:
        self._action = action.iloc[0].to_numpy()

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
        observation, _info = self._env.reset(seed=self._cfg.seed)

        self._last_step = StepData(
            observation=observation,
            reward=np.nan,
            action=np.full(self._action_shape, 0),
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
        obs_data = {}
        for i in range(len(step.observation)):
            obs_data[f'tag-{i}'] = step.observation[i]

        action_data = {}
        for i in range(len(step.action)):
            action_data[f'action-{i}'] = step.action[i]

        meta_data = {
            "reward": step.reward,
            "truncated": step.truncated,
            "terminated": step.terminated,
        }

        # mash all columns together to simulate tags from OPC
        tags = obs_data | action_data | meta_data
        # filter for only the tags specified in the tag configs
        tags = { k: v for k, v in tags.items() if k in self._all_tag_names }

        idx = pd.DatetimeIndex([self.clock])
        df = pd.DataFrame(tags, index=idx)
        return df
