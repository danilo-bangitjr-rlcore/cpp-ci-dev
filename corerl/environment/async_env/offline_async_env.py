import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import datetime, timedelta, UTC
from corerl.configs.config import config, MISSING
from dataclasses import dataclass
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, StepData
from corerl.utils.gym import space_bounds
from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig

import yaml


@config()
class OfflineAsyncEnvConfig:
    name: str = "offline_async_env"
    seed: int = 0
    discrete_control: bool = False
    # gym_name: str = MISSING
    bucket_width: str = MISSING
    aggregation: str = MISSING
    env_start_time: str = MISSING
    env_end_time: str = MISSING
    env_step_time: str = MISSING
    db: TagDBConfig = MISSING


class OfflineAsyncEnv(AsyncEnv):
    def __init__(self, cfg: OfflineAsyncEnvConfig, tags: list[TagConfig]):

        self.bucket_width = pd.Timedelta(cfg.bucket_width).to_pytimedelta()
        self.env_step_time = pd.Timedelta(cfg.env_step_time).to_pytimedelta()
        self._data_reader = DataReader(db_cfg=cfg.db)

        if cfg.env_start_time == MISSING or cfg.env_end_time == MISSING:
            time_stats = self._data_reader.get_time_stats()

        if cfg.env_start_time == MISSING:
            self.env_start_time = time_stats.start
        else:
            self.env_start_time = datetime.strptime(
                cfg.env_start_time,
                '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)

        if cfg.env_end_time == MISSING:
            self.env_end_time = time_stats.end
        else:
            self.env_end_time = datetime.strptime(
                cfg.env_end_time,
                '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)

        print(self.env_start_time, self.env_end_time)
        self.current_start_time = self.env_start_time

        self.names = [
            tag.name for tag in tags
        ]

        self._action: np.ndarray | None = None
        self._last_step: StepData | None = None

    def emit_action(self, action: np.ndarray) -> None:
        self._action = action

    def get_latest_obs(self) -> pd.DataFrame:
        obs = self._data_reader.batch_aggregated_read(
            self.names,
            self.current_start_time,
            self.current_start_time + self.env_step_time,
            bucket_width=self.bucket_width,
        )

        self.current_start_time += self.env_step_time
        if self.current_start_time >= self.env_end_time:
            self.current_start_time = self.env_start_time

        # increment time
        # if its finished set term to true
        # reset
        return obs

    def _reset():
        # Reset the counter
        # Return first obs
        ...



