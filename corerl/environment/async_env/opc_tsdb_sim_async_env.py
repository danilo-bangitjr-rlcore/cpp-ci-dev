from datetime import UTC, datetime
from time import sleep

import numpy as np
import pandas as pd

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, BaseAsyncEnvConfig


@config()
class OPCTSDBSimAsyncEnvConfig(BaseAsyncEnvConfig):
    name: str = "opc_tsdb_sim_async_env"
    db: TagDBConfig = MISSING
    bucket_width: str = MISSING
    opc_conn_url: str = MISSING


class OPCTSDBSimAsyncEnv(AsyncEnv):
    """The OPC TSDB Sim Async Env exposes a mechanism to interact with a farama gym environment using OPC to represent
    writing actions and TSDB to read observations/state.
    """

    def __init__(self, cfg: OPCTSDBSimAsyncEnvConfig, tags: list[TagConfig]):
        pd_bucket_width = pd.Timedelta(cfg.bucket_width)
        assert isinstance(pd_bucket_width, pd.Timedelta), "Failed parsing of bucket_width"
        self.bucket_width = pd_bucket_width.to_pytimedelta()

        self._data_reader = DataReader(db_cfg=cfg.db)

        self.env_start_time = datetime.now(UTC)
        self.current_start_time = self.env_start_time

        self.names = [
            tag.name for tag in tags
        ]

        self.obs_names = [
            tag.name for tag in tags if not tag.is_action and not tag.is_meta
        ]

        self.action_names = [
            tag.name for tag in tags if tag.is_action
        ]

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
            read_start = read_start - self.bucket_width

        res = self._data_reader.batch_aggregated_read(
            self.names,
            read_start,
            self.current_start_time + self.bucket_width,
            self.bucket_width
        )
        while res.isnull().values.any():
            # sleep until all batch aggregated read values are defined (e.g. running simulation, sensor data)
            sleep(2)
            res = self._data_reader.batch_aggregated_read(
                self.obs_names + self.action_names,
                read_start,
                self.current_start_time + self.bucket_width,
                self.bucket_width
            )

        self.current_start_time += self.bucket_width
        return res
