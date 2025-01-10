from datetime import UTC, datetime

import numpy as np
import pandas as pd
from asyncua.sync import Client
from asyncua.ua.uatypes import VariantType

from corerl.configs.config import config

# Data Pipline
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv, OPCEnvConfig, TSDBEnvConfig


@config()
class DepAsyncEnvConfig(TSDBEnvConfig, OPCEnvConfig):
    name: str = "dep_async_env"
    sleep_sec: int = 1


class DeploymentAsyncEnv(AsyncEnv):
    """
    It's going to be sync for now
    """

    def __init__(self, cfg: DepAsyncEnvConfig, tags: list[TagConfig]):
        self.url = cfg.opc_conn_url
        self.ns = cfg.opc_ns

        self.tags = tags
        self.obs_period = cfg.obs_period

        self.action_tags = [tag for tag in self.tags if tag.is_action]
        self.observation_tags = [tag for tag in self.tags if not tag.is_action and not tag.is_meta]

        self.client = Client(self.url)
        self.client.connect()

        self.data_reader = DataReader(db_cfg=cfg.db)


    def _make_opc_node_id(self, str_id: str, namespace: int = 0):
        return f"ns={namespace};s={str_id}"

    def close(self):
        """Closes the opc client and data reader
        Can also use __exit__ or cleanup
        """
        self.client.disconnect()
        self.data_reader.close()

    def emit_action(self, action: np.ndarray) -> None:
        """Writes directly to the OPC server"""
        for action_val, action_tag in zip(action, self.action_tags, strict=True):
            node = self.client.get_node(self._make_opc_node_id(action_tag.name, namespace=self.ns))
            node.write_value(action_val, VariantType.Double)  # Assuming that all are doubles

    def get_latest_obs(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        end_time = now
        start_time = end_time - self.obs_period
        sensor_names = [tag.name for tag in self.observation_tags]
        return self.data_reader.single_aggregated_read(sensor_names, start_time=start_time, end_time=end_time)
