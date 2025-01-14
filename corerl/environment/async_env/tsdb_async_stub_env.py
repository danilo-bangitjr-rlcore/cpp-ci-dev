from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.db.data_reader import DataReader, TimeStats
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, GymEnvConfig, TSDBEnvConfig
from corerl.environment.reward.scrubber import ScrubberReward, ScrubberRewardConfig


@config()
class TSDBAsyncStubEnvConfig(TSDBEnvConfig, GymEnvConfig):
    name: str = "tsdb_async_stub_env"
    env_start_time: str = MISSING
    env_end_time: str = MISSING


class TSDBAsyncStubEnv(AsyncEnv):
    """The TSDBAsyncStubEnv is a sanity check environment that only serves to verify that our pipeline can consume data
    from our timescale database without exception. **It is not intended to train any models.**


    TODO:
        * Build an async environment such that the `emit_action` call writes to OPC and the timescale database is live
          updated with observations from the environment.
    """

    def __init__(self, cfg: TSDBAsyncStubEnvConfig, tags: list[TagConfig]):
        self.gym_name = cfg.gym_name

        self.env_step_time = cfg.obs_period
        self.bucket_width = cfg.obs_period

        self._data_reader = DataReader(db_cfg=cfg.db)

        # Getting defaults for start and end time
        time_stats: TimeStats | None = None
        if cfg.env_start_time == MISSING or cfg.env_end_time == MISSING:
            time_stats = self._data_reader.get_time_stats()

        if cfg.env_start_time == MISSING:
            assert isinstance(time_stats, TimeStats), "Error getting time stats"
            self.env_start_time = time_stats.start
        else:
            self.env_start_time = datetime.strptime(
                cfg.env_start_time,
                '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)

        if cfg.env_end_time == MISSING:
            assert isinstance(time_stats, TimeStats), "Error getting time stats"
            self.env_end_time = time_stats.end
        else:
            self.env_end_time = datetime.strptime(
                cfg.env_end_time,
                '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)

        self.current_start_time = self.env_start_time

        self.names = [
            tag.name for tag in tags
        ]

        self.obs_names = [
            tag.name for tag in tags
            if not tag.is_meta and tag.action_constructor is not None
        ]

        self.action_names = [
            tag.name for tag in tags
            if tag.action_constructor is not None
        ]

        # This could come from a corerl reward function factory
        self._reward_func = self._init_reward()

    def emit_action(self, action: np.ndarray) -> None:
        """
        Because this environment is intended to verify correctness of the interaction
        between the TSDB data source and our pipeline and not intended to train an agent
        emiting an action here is a no-op.
        """
        pass

    def get_latest_obs(self) -> pd.DataFrame:
        read_start = self.current_start_time
        if read_start > self.env_start_time:
            # temporal state pipeline logic requires last step's latest time bucket
            read_start = read_start - self.env_step_time

        res = self._data_reader.single_aggregated_read(
            self.obs_names + self.action_names,
            read_start,
            self.current_start_time + self.env_step_time - timedelta(microseconds=1),
        )

        self.current_start_time += self.env_step_time
        term = False

        if self.current_start_time >= self.env_end_time:
            self.current_start_time = self.env_start_time
            term = True

        res = res.assign(reward = self._reward_func(res))
        res = res.assign(truncated = False)
        res = res.assign(terminated = term)

        return res

    def _init_reward(self):
        match self.gym_name:
            case "epcor_tsdb_scrubber":
                reward_cfg = ScrubberRewardConfig(
                    reward_tags = None,
                    steps_per_interaction = 1,
                    only_dp_transitions = False,
                    type = 'mean_efficiency_cost',
                )
                scrubber_reward = ScrubberReward(reward_cfg)

                # This wrapper is for calling scrubber_reward in a simpler manner
                def wrapped_scrubber_reward(res: pd.DataFrame):
                    return scrubber_reward(np.ndarray(0), df_rows=res)

                return wrapped_scrubber_reward
            case _:
                raise NotImplementedError
