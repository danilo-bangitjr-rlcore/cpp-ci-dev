import numpy as np
import pandas as pd
from datetime import datetime, UTC

from corerl.configs.config import config, MISSING
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig, TimeStats
from corerl.environment.reward.scrubber import ScrubberReward, ScrubberRewardConfig


@config()
class OfflineAsyncEnvConfig:
    name: str = "offline_async_env"
    seed: int = 0
    discrete_control: bool = False
    gym_name: str = MISSING
    aggregation: str = MISSING
    env_start_time: str = MISSING
    env_end_time: str = MISSING
    env_step_time: str = MISSING
    db: TagDBConfig = MISSING
    bucket_width: str = MISSING


class OfflineAsyncEnv(AsyncEnv):
    def __init__(self, cfg: OfflineAsyncEnvConfig, tags: list[TagConfig]):
        self.gym_name = cfg.gym_name

        pd_env_step_time = pd.Timedelta(cfg.env_step_time)
        assert isinstance(pd_env_step_time, pd.Timedelta), "Failed parsing of env_step_time"
        self.env_step_time = pd_env_step_time.to_pytimedelta()

        pd_bucket_width = pd.Timedelta(cfg.bucket_width)
        assert isinstance(pd_bucket_width, pd.Timedelta), "Failed parsing of bucket_width"
        self.bucket_width = pd_bucket_width.to_pytimedelta()

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
            tag.name for tag in tags if not tag.is_action
        ]

        self.action_names = [
            tag.name for tag in tags if tag.is_action
        ]

        # This could come from a corerl reward function factory
        self._reward_func = self._init_reward()

    def emit_action(self, action: np.ndarray) -> None:
        pass

    def get_latest_obs(self) -> pd.DataFrame:
        read_start = self.current_start_time
        if read_start >= self.env_start_time:
            # temporal state pipeline logic requires last step's latest time bucket
            read_start = read_start - self.env_step_time

        res = self._data_reader.batch_aggregated_read(
            self.names,
            read_start,
            self.current_start_time + self.env_step_time,
            self.bucket_width
        )

        self.current_start_time += self.env_step_time
        term = False

        if self.current_start_time >= self.env_end_time:
            self.current_start_time = self.env_start_time
            term = True

        res["reward"] = self._reward_func(res)
        res["trunc"] = 0.0
        res["term"] = term

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
