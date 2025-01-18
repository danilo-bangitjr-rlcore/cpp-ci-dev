import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from asyncua.sync import Client

from corerl.configs.config import MISSING, config

# Data Pipline
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import NormalizerConfig
from corerl.environment.async_env.async_env import AsyncEnv, OPCEnvConfig, TSDBEnvConfig
from corerl.utils.list import find_instance
from corerl.utils.maybe import Maybe
from corerl.utils.opc_connection import make_opc_node_id

logger = logging.getLogger(__file__)

@config()
class DepAsyncEnvConfig(TSDBEnvConfig, OPCEnvConfig):
    name: str = "dep_async_env"
    action_tolerance: timedelta = MISSING


class DeploymentAsyncEnv(AsyncEnv):
    """
    It's going to be sync for now
    """

    def __init__(self, cfg: DepAsyncEnvConfig, tag_configs: list[TagConfig]):
        self.url = cfg.opc_conn_url
        self.ns = cfg.opc_ns

        self.tag_configs = tag_configs
        self.obs_period = cfg.obs_period
        self.action_period = cfg.action_period
        self.action_tolerance = cfg.action_tolerance

        self.tag_names = [tag.name for tag in tag_configs]
        self._action_tags = [tag for tag in tag_configs if tag.action_constructor is not None]
        self._meta_tags = [tag for tag in tag_configs if tag.is_meta]
        self._observation_tags = [
            tag for tag in tag_configs
            if not tag.is_meta and tag.action_constructor is None
        ]

        self._opc_client = Client(self.url)
        self._opc_client.connect()

        self.data_reader = DataReader(db_cfg=cfg.db)

        # define opc action nodes
        self.action_nodes = []
        for tag in tag_configs:
            if tag.action_constructor is None:
                continue

            node_name = tag.name
            if tag.node_identifier is not None:
                node_name = tag.node_identifier

            id = make_opc_node_id(node_name, cfg.opc_ns)
            node = self._opc_client.get_node(id)
            self.action_nodes.append(node)


    def _make_opc_node_id(self, str_id: str, namespace: int = 0):
        return f"ns={namespace};s={str_id}"

    def close(self):
        """Closes the opc client and data reader
        Can also use __exit__ or cleanup
        """
        self._opc_client.disconnect()
        self.data_reader.close()

    def emit_action(self, action: np.ndarray) -> None:
        """Writes directly to the OPC server"""

        denormalized_actions = self._denormalize_action(action)
        action_names = [tag.name for tag in self._action_tags]
        logger.info(f"emitting actions {action_names} with values {denormalized_actions}...")
        self._opc_client.write_values(self.action_nodes, denormalized_actions)

    def _denormalize_action(self, action: np.ndarray) -> list[float]:
        denormalized_actions = []
        action_tag_configs = self._action_tags
        action = action.flatten()
        assert len(action) == len(action_tag_configs)
        action_dim = len(action)

        for act_i in range(action_dim):
            # denormalize the action if possible, otherwise emit normalized action
            raw_action = action[act_i]
            action_cfg = action_tag_configs[act_i]

            norm_cfg = (
                Maybe(action_cfg.action_constructor)
                .map(lambda ac: find_instance(NormalizerConfig, ac))
            )

            lo = (
                norm_cfg
                .map(lambda cfg: cfg.min)
                .or_else(0)
            )

            hi = (
                norm_cfg
                .map(lambda cfg: cfg.max)
                .or_else(1)
            )

            scale = hi - lo
            bias = lo
            denormalized_actions.append(scale * raw_action + bias)

        return denormalized_actions

    def get_latest_obs(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        obs = self.data_reader.single_aggregated_read(
            names=self.tag_names,
            start_time=now - self.obs_period,
            end_time=now
        )
        return obs
