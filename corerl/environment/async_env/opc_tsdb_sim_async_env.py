import logging
from datetime import UTC, datetime, timedelta
from time import sleep

import numpy as np
import pandas as pd
from asyncua.sync import Client

from corerl.configs.config import config
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, OPCEnvConfig, GymEnvConfig, TSDBEnvConfig
from corerl.utils.opc_connection import make_opc_node_id

log = logging.getLogger(__name__)


@config()
class OPCTSDBSimAsyncEnvConfig(GymEnvConfig, OPCEnvConfig, TSDBEnvConfig):
    name: str = "opc_tsdb_sim_async_env"
    sleep_sec: int = 10
    obs_fetch_attempts: int = 20
    obs_read_delay_buffer: timedelta = timedelta(milliseconds=50)


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
        self.obs_period = cfg.obs_period
        self.obs_read_delay_buffer = cfg.obs_read_delay_buffer

        self._data_reader = DataReader(db_cfg=cfg.db)
        self.obs_fetch_attempts = cfg.obs_fetch_attempts

        self.env_start_time = datetime.now(UTC)
        self.current_start_time = self.env_start_time

        self.tag_configs = tag_configs

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
        denormalized_actions = []
        action_tag_configs = [tag for tag in self.tag_configs if tag.is_action]
        assert len(action.flatten()) == len(action_tag_configs)

        for act_i in range(len(action.flatten())):
            # denormalize the action if possible, otherwise emit normalized action
            raw_action = action.flatten()[act_i]
            try:
                action_tag_config = action_tag_configs[act_i]
                lo, hi = action_tag_config.bounds
                assert isinstance(lo, float) and isinstance(hi, float)
                scale = hi - lo
                bias = lo
                denormalized_actions.append(scale * raw_action + bias)
            except AssertionError:
                denormalized_actions.append(raw_action)
        self._opc_client.write_values(self.action_nodes, denormalized_actions)

    def get_latest_obs(self) -> pd.DataFrame:
        read_start = self.current_start_time
        if read_start > self.env_start_time:
            # temporal state pipeline logic requires last step's latest time bucket
            read_start = read_start - self.obs_period

        read_end = self.current_start_time + self.obs_period - timedelta(microseconds=1)

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
        if now <= (read_end + self.obs_read_delay_buffer):
            # the query end time is in the future, wait until this has elapsed before requesting observations from TSDB
            wait_time_delta = (read_end - now) + self.obs_read_delay_buffer
            sleep(wait_time_delta.total_seconds())

        res = get_obs_df()
        self.current_start_time += self.obs_period
        return res

    def cleanup(self):
        """
        Close the OPC client and datareader sql connection
        """
        self._data_reader.close()
        self._opc_client.disconnect()
