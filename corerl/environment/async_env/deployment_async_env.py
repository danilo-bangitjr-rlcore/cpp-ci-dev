import numpy as np
import pandas as pd
import gymnasium as gym

from datetime import UTC, datetime, timedelta

from asyncua.sync import Client
from asyncua.ua.uatypes import VariantType

from corerl.environment.async_env.async_env import AsyncEnv

from corerl.utils.gymnasium import gen_tag_configs_from_env

# Data Pipline
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_reader import TagDBConfig


class DeploymentAsyncEnv(AsyncEnv):
    '''
    It's going to be sync for now
    '''
    def __init__(self, cfg):
        self.url = "opc.tcp://admin@0.0.0.0:4840/rlcore/server/"

        self.ns = 2
        if "ns" in cfg:
            self.ns = cfg.ns

        if 'environment' in cfg:
            match cfg.environment.type:
                case 'gym.make':
                    # A little cheat to get the tags, hopefully in the future we have a more robust method
                    env = gym.make(*cfg.environment.args, **cfg.environment.kwargs)
                    self.tags = gen_tag_configs_from_env(env)
                case _:
                    raise NotImplementedError
        else:
            raise KeyError("Configuration needs cfg.environment")

        self.timedelta = cfg.timedelta if "timedelta" in cfg else 1 # In minutes

        self.action_tags = self.tags["action"]
        self.observation_tags = self.tags["observation"]

        self.client = Client(self.url)
        self.client.connect()

        db_cfg = TagDBConfig(
            drivername="postgresql+psycopg2",
            username="postgres",
            password="password",
            ip="localhost",
            port=5432,  # default is 5432, but we want to use different port for test db
            db_name="postgres",
            sensor_table_name="public.opcua",
        )

        self.data_reader = DataReader(db_cfg=db_cfg)


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
        sensor_names = [f"{tag.name}" for tag in self.observation_tags]
        return self.data_reader.single_aggregated_read(sensor_names, start_time=start_time, end_time=end_time)

