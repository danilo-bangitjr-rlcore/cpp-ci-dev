import logging
from datetime import UTC, datetime
from time import sleep

import numpy as np
import pandas as pd
from asyncua.sync import Client

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, BaseAsyncEnvConfig
from corerl.utils.opc_connection import make_opc_node_id

log = logging.getLogger(__name__)

@config()
class OPCTSDBSimAsyncEnvConfig(BaseAsyncEnvConfig):
    name: str = "opc_tsdb_sim_async_env"
    db: TagDBConfig = MISSING
    bucket_width: str = MISSING
    opc_conn_url: str = MISSING
    opc_ns: int = MISSING  # OPC node namespace, this is almost always going to be `2`
    sleep_sec: int = 10
    obs_fetch_attempts: int = 20

class OPCTSDBSimAsyncEnv(AsyncEnv):
    """The OPC TSDB Sim Async Env exposes a mechanism to interact with a farama gym environment using OPC to represent
    writing actions and TSDB to read observations/state.

    This environment may be used with config **env.name: opc_tsdb_sim_async_env**.

    1. Create a new config with a farama gym environment within **env.gym_name** that contains continuous actions.
    2. Run **e2e/make_configs.py** with the configuration to generate the telegraf and OPC yaml tag configs.
    3. Add the generated yaml tag configs stub to the new config.
    4. Run **docker compose up** to create the OPC server, postgres DB, and configured telegraf service.
    5. Run **e2e/opc_client.py** with the configuration to start the farama gym environment.
    6. Run **main.py** with the configuration to start the agent that communicates using TSDB/OPC.

    """

    def __init__(self, cfg: OPCTSDBSimAsyncEnvConfig, tag_configs: list[TagConfig]):
        pd_bucket_width = pd.Timedelta(cfg.bucket_width)
        assert isinstance(pd_bucket_width, pd.Timedelta), "Failed parsing of bucket_width"
        self.bucket_width = pd_bucket_width.to_pytimedelta()

        self._data_reader = DataReader(db_cfg=cfg.db)
        self.obs_fetch_attempts = cfg.obs_fetch_attempts

        self.env_start_time = datetime.now(UTC)
        self.current_start_time = self.env_start_time

        self.names = [tag.name for tag in tag_configs]

        self.obs_names = [tag.name for tag in tag_configs if not tag.is_action and not tag.is_meta]
        self.action_names = [tag.name for tag in tag_configs if tag.is_action]
        self.meta_names = [tag.name for tag in tag_configs if tag.is_meta]

        self._opc_client = Client(cfg.opc_conn_url)
        self._opc_client.connect()

        # define opc action nodes
        self.action_nodes = []
        for tag in tag_configs:
            if not tag.is_action:
                continue
            id = make_opc_node_id(tag.name, cfg.opc_ns)
            node = self._opc_client.get_node(id)
            self.action_nodes.append(node)

    def emit_action(self, action: np.ndarray) -> None:
        self._opc_client.write_values(self.action_nodes, action.tolist())

    def get_latest_obs(self) -> pd.DataFrame:
        read_start = self.current_start_time
        if read_start > self.env_start_time:
            # temporal state pipeline logic requires last step's latest time bucket
            read_start = read_start - self.bucket_width

        read_end = self.current_start_time + self.bucket_width - timedelta(microseconds=1)

        def get_obs_df():
            act_obs_reward = self._data_reader.single_aggregated_read(
                self.action_names + self.obs_names + ["reward"],
                read_start,
                read_end,
            )
            meta = self._data_reader.single_aggregated_read(
                [name for name in self.meta_names if name != "reward"],
                read_start,
                read_end,
                "bool_or",
            )
            return pd.concat([act_obs_reward, meta], axis=1)

        now = datetime.now(UTC)
        if now <= read_end:
            # the query end time is in the future, wait until this has elapsed before requesting observations from TSDB
            wait_time_delta = read_end - now
            sleep(wait_time_delta.total_seconds())

        res = get_obs_df()
        self.current_start_time += self.bucket_width
        return res

    def cleanup(self):
        """
        Close the OPC client and datareader sql connection
        """
        self._data_reader.close()
        self._opc_client.disconnect()
