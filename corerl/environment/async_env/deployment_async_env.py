from dataclasses import field
from datetime import UTC, datetime, timedelta

import gymnasium as gym
import numpy as np
import pandas as pd
from asyncua.sync import Client
from asyncua.ua.uatypes import VariantType

from corerl.configs.config import config

# Data Pipline
from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.environment.config import EnvironmentConfig
from corerl.utils.gymnasium import gen_tag_configs_from_env


@config()
class DepAsyncEnvConfig(EnvironmentConfig):
    ns: int = 2
    timedelta: int = 1
    sleep_sec: int = 1

    opc_url: str = "opc.tcp://admin@0.0.0.0:4840/rlcore/server/"

    db: TagDBConfig = field(
        default_factory=lambda: TagDBConfig(
            db_name="postgres",
            sensor_table_name="opcua",
        )
    )


class DeploymentAsyncEnv(AsyncEnv):
    '''
    It's going to be sync for now
    '''
    def __init__(self, cfg: DepAsyncEnvConfig):
        self.url = "opc.tcp://admin@0.0.0.0:4840/rlcore/server/"

        self.ns = cfg.ns

        match cfg.type:
            case 'gym.make':
                # A little cheat to get the tags, hopefully in the future we have a more robust method
                env = gym.make(cfg.name)
                self.tags = gen_tag_configs_from_env(env)
            case _:
                raise NotImplementedError

        self.timedelta = cfg.timedelta # in minutes

        self.action_tags = [tag for tag in self.tags if tag.tag_type == "action"]
        self.observation_tags = [tag for tag in self.tags if tag.tag_type == "observation"]

        self.client = Client(self.url)
        self.client.connect()

        self.data_reader = DataReader(db_cfg=cfg.db)


    def _make_opc_node_id(self, str_id: str, namespace=0):
        return f"ns={namespace};s={str_id}"

    def close(self):
        '''Closes the opc client and data reader
        Can also use __exit__ or cleanup
        '''
        self.client.disconnect()
        self.data_reader.close()


    def emit_action(self, action: np.ndarray) -> None:
        '''Writes directly to the OPC server'''
        for action_val, action_tag in zip(action, self.action_tags, strict=True):
            node = self.client.get_node(self._make_opc_node_id(action_tag.name, namespace=self.ns))
            node.write_value(action_val, VariantType.Double)  # Assuming that all are doubles

    def get_latest_obs(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        end_time = now
        start_time = end_time - timedelta(minutes=self.timedelta)
        sensor_names = [tag.name for tag in self.observation_tags]
        return self.data_reader.single_aggregated_read(sensor_names, start_time=start_time, end_time=end_time)
