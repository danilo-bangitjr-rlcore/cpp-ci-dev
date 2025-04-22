import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import SupportsFloat

import numpy as np
import pandas as pd

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import TagDBConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.environment.factory import init_environment
from corerl.utils.coreio import CoreIOThinClient
from corerl.utils.gym import space_bounds, space_shape

logger = logging.getLogger(__name__)

class DummyThinClient(CoreIOThinClient):
    def __init__(self, coreio_origin: str):
        ...

class DummyDataReader(DataReader):
    def __init__(self, db_cfg: TagDBConfig) -> None:
        ...

@dataclass
class StepData:
    observation: np.ndarray
    reward: SupportsFloat
    action: np.ndarray
    truncated: bool
    terminated: bool


class SimAsyncEnv(DeploymentAsyncEnv):
    """AsyncEnv which directly runs and steps through a Farama Gymnasium environment.
    """
    def __init__(self, cfg: AsyncEnvConfig, tag_configs: list[TagConfig]):
        super().__init__(cfg, tag_configs)

        ### simulation specific initialization ###
        assert self._cfg.gym is not None
        self._sim_cfg = self._cfg.gym
        self._env = init_environment(self._cfg.gym)

        shape = self._env.observation_space.shape
        assert shape is not None
        assert len(shape) == 1, "Cannot handle environments with non-vector observations"

        self._action_bounds = space_bounds(self._env.action_space)
        self._action_shape = space_shape(self._env.action_space)

        self._all_tag_names = {tag.name for tag in tag_configs}

        self.clock = datetime(1984, 1, 1, tzinfo=UTC)
        self._clock_inc = cfg.obs_period

        self._action: np.ndarray | None = None
        self._last_step: StepData | None = None

    def _init_thinclient(self):
        return DummyThinClient(self._cfg.coreio_origin)

    def _init_datareader(self):
        return DummyDataReader(self._cfg.db)

    def _register_action_nodes(self):
        ...

    # ------------------
    # -- AsyncEnv API --
    # ------------------
    def emit_action(self, action: pd.DataFrame, log_action: bool = False) -> None:
        if log_action:
            logger.info("--- Emitting action ---")
            for line in action.to_string().splitlines():
                logger.info(line)

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
        observation, _info = self._env.reset(seed=self._sim_cfg.seed)

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
        for i, obs in enumerate(step.observation):
            obs_data[f'tag-{i}'] = obs

        action_data = {}
        for i, act in enumerate(step.action):
            action_data[f'action-{i}'] = act

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
