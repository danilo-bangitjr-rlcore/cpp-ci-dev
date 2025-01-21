import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from asyncua.sync import Client

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, GymEnvConfig, OPCEnvConfig, TSDBEnvConfig
from corerl.utils.opc_connection import make_opc_node_id

log = logging.getLogger(__name__)


@config()
class OPCTSDBSimAsyncEnvConfig(GymEnvConfig, OPCEnvConfig, TSDBEnvConfig):
    name: str = "opc_tsdb_sim_async_env"
    action_tolerance: timedelta = MISSING
    obs_fetch_attempts: int = 20
    # obs_read_delay_buffer: timedelta = timedelta(seconds=1)


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
        self.action_period = cfg.action_period
        # self.obs_read_delay_buffer = cfg.obs_read_delay_buffer
        self.action_tolerance = cfg.action_tolerance

        self.data_reader = DataReader(db_cfg=cfg.db)
        self.obs_fetch_attempts = cfg.obs_fetch_attempts

        self.env_start_time = datetime.now(UTC)
        self.env_start_time = self.env_start_time.replace(microsecond=0)
        self.tag_configs = tag_configs

        self._action_tags = [tag for tag in tag_configs if tag.action_constructor is not None]
        self._meta_tags = [tag for tag in tag_configs if tag.is_meta]
        self._obs_tags = [
            tag for tag in tag_configs
            if not tag.is_meta and tag.action_constructor is None
        ]

        self._opc_client = Client(cfg.opc_conn_url)
        self._opc_client.connect()

        # define opc action nodes
        self.action_nodes = []
        for tag in tag_configs:
            if tag.action_constructor is None:
                continue
            id = make_opc_node_id(tag.name, cfg.opc_ns)
            node = self._opc_client.get_node(id)
            self.action_nodes.append(node)

    def emit_action(self, action: np.ndarray) -> None:
        denormalized_actions = []
        assert len(action.flatten()) == len(self._action_tags)

        for act_i in range(len(action.flatten())):
            # denormalize the action if possible, otherwise emit normalized action
            raw_action = action.flatten()[act_i]
            try:
                action_tag_config = self._action_tags[act_i]
                assert action_tag_config.operating_range is not None
                lo, hi = action_tag_config.operating_range
                assert isinstance(lo, float) and isinstance(hi, float)
                scale = hi - lo
                bias = lo
                denormalized_actions.append(scale * raw_action + bias)
            except AssertionError:
                denormalized_actions.append(raw_action)
        self._opc_client.write_values(self.action_nodes, denormalized_actions)

    def get_latest_obs(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        read_start = now - self.obs_period
        read_end = now

        def get_obs_df():
            action_names = [tag.name for tag in self._action_tags]
            obs_names = [tag.name for tag in self._obs_tags]
            meta_names = [tag.name for tag in self._meta_tags]

            act_obs_reward = self.data_reader.single_aggregated_read(
                action_names + obs_names + ["gym_reward"],
                read_start,
                read_end,
            )
            meta = self.data_reader.single_aggregated_read(
                [name for name in meta_names if name != "gym_reward"],
                read_start,
                read_end,
                "bool_or",
            )
            return pd.concat([act_obs_reward, meta], axis=1)

        res = get_obs_df()
        return res

    def cleanup(self):
        """
        Close the OPC client and datareader sql connection
        """
        self.data_reader.close()
        self._opc_client.disconnect()
