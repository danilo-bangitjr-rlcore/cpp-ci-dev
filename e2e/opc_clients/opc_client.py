# instantiate our OPC client
# inttantiate a farama gym environment instance
# all gym environment actions have side effects that interact with OPC client
# all agent observations are performed through timescale DB

import logging
from dataclasses import field
from time import sleep

import gymnasium as gym
import numpy as np
from asyncua.sync import Client, SyncNode
from asyncua.ua.uaerrors import BadNodeIdExists, BadNodeIdUnknown
from asyncua.ua.uatypes import VariantType

from corerl.configs.config import MISSING, config
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.environment.async_env.opc_tsdb_sim_async_env import OPCTSDBSimAsyncEnvConfig
from corerl.environment.factory import init_environment
from corerl.utils.opc_connection import make_opc_node_id


@config(allow_extra=True)
class Config:
    env: AsyncEnvConfig = MISSING
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def initialize_opc_folder(client: Client, cfg_env: OPCTSDBSimAsyncEnvConfig):
    # create folder containing environment variables
    folder_node_id = make_opc_node_id(cfg_env.name)
    try:
        folder = client.nodes.objects.add_folder(folder_node_id, cfg_env.name)
    except BadNodeIdExists:
        # folder already exists
        folder = client.get_node(folder_node_id)
    return folder


def initialize_opc_nodes_from_tags(
    client: Client,
    cfg_env: OPCTSDBSimAsyncEnvConfig,
    tag_configs: list[TagConfig],
    initial_observation: np.ndarray,
    initial_action: np.ndarray,
):
    folder = initialize_opc_folder(client, cfg_env)
    # create OPC nodes based on tags
    opc_nodes: dict[str, list[SyncNode]] = {
        "action": [],
        "observation": [],
        "meta": [],
    }

    action_idx = 0
    observation_idx = 0
    for tag in tag_configs:
        if tag.is_action:
            tag_type = "action"
        elif not tag.is_action and not tag.is_meta:
            tag_type = "observation"
        elif tag.is_meta:
            tag_type = "meta"
        else:
            raise RuntimeError("Invalid tag provided", tag)

        id = make_opc_node_id(tag.name, cfg_env.opc_ns)
        node = client.get_node(id)

        try:
            _ = node.read_browse_name()
        except BadNodeIdUnknown:
            # node does not exist in OPC server, create it
            # instantiate first action as random sample, store in OPC
            val = 0.0
            var_type = VariantType.Double
            if tag_type == "action":
                val = initial_action[action_idx]
                action_idx += 1
            elif tag_type == "observation":
                val = initial_observation[observation_idx]
                observation_idx += 1
            elif tag_type == "meta":
                if tag.name == "reward":
                    val = 0.0
                elif tag.name == "truncated":
                    val = False
                    var_type = VariantType.Boolean
                elif tag.name == "terminated":
                    val = False
                    var_type = VariantType.Boolean
            node = folder.add_variable(id, tag.name, val, var_type)

        opc_nodes[tag_type].append(node)
    return opc_nodes


def run(env: gym.Env, client: Client, cfg_env: OPCTSDBSimAsyncEnvConfig, tag_configs: list[TagConfig]):
    seed = cfg_env.seed
    sleep_sec = cfg_env.sleep_sec

    initial_observation, info = env.reset(seed=seed)
    initial_action = env.action_space.sample()

    opc_nodes = initialize_opc_nodes_from_tags(client, cfg_env, tag_configs, initial_observation, initial_action)

    # Run env forever using OPC for actions, observations, and rewards
    step_counter = 0
    while True:
        # get the action values from OPC
        action_values = client.read_values(opc_nodes["action"])

        # read the observation from the environment, write to OPC
        observation, reward, terminated, truncated, info = env.step(action_values)
        if terminated or truncated:
            observation, info = env.reset()

        # write the observation values to OPC
        client.write_values(opc_nodes["observation"], observation.tolist())

        # write the reward to OPC
        client.write_values(opc_nodes["meta"], [reward, terminated, truncated])

        if sleep_sec:
            _logger.info(f"Sleeping for {sleep_sec}s, step counter: {step_counter}")
            sleep(sleep_sec)
        step_counter += 1


@load_config(Config, base="config/")
def main(cfg: Config):
    assert isinstance(cfg.env, OPCTSDBSimAsyncEnvConfig), "opc client sim only supported for OPCTSDBSimAsyncEnvConfig"
    env: gym.Env = init_environment(cfg.env)
    _logger.info(f"Running OPC env simulation {env}")

    client = Client(cfg.env.opc_conn_url)
    try:
        client.connect()
        run(env, client, cfg.env, cfg.pipeline.tags)
    except Exception as e:
        _logger.exception(e)
    finally:
        client.disconnect()
        env.close()


if __name__ == "__main__":
    _logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", encoding="utf-8", level=logging.INFO)
    main()
