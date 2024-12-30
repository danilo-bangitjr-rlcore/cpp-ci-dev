# instantiate our OPC client
# inttantiate a farama gym environment instance
# instantiate a 'dumb' agent (e.g. random actions)
# all gym environment actions have side effects that interact with OPC client
# all agent observations are performed through timescale DB

from dataclasses import field
import logging
from time import sleep

import gymnasium as gym
from asyncua.sync import Client, SyncNode
from asyncua.ua.uaerrors import BadNodeIdExists, BadNodeIdUnknown
from asyncua.ua.uatypes import VariantType

from corerl.configs.config import config
from corerl.configs.loader import load_config
from corerl.environment.async_env.deployment_async_env import DepAsyncEnvConfig
from corerl.environment.factory import init_environment
from corerl.utils.gymnasium import gen_tag_configs_from_env


@config(allow_extra=True)
class Config:
    env: DepAsyncEnvConfig = field(default_factory=DepAsyncEnvConfig)


def make_opc_node_id(str_id: str, namespace: int = 0):
    return f"ns={namespace};s={str_id}"


def run(env: gym.Env, client: Client, cfg: Config):
    seed = cfg.env.seed
    ns = cfg.env.ns
    sleep_sec = cfg.env.sleep_sec

    # create folder containing environment variables
    folder_node_id = make_opc_node_id(cfg.env.name)
    try:
        folder = client.nodes.objects.add_folder(folder_node_id, cfg.env.name)
    except BadNodeIdExists:
        # folder already exists
        folder = client.get_node(folder_node_id)

    tag_configs = gen_tag_configs_from_env(env)

    # create OPC nodes based on tags
    opc_nodes: dict[str, list[SyncNode]] = {}

    initial_observation, info = env.reset(seed=seed)
    initial_action = env.action_space.sample()

    # for key, tags in tag_configs.items():
    for tag_idx, tag in enumerate(tag_configs):
        tag_type = "observation"
        if tag.is_action:
            tag_type = "action"
        elif tag.is_meta:
            tag_type = "meta"

        opc_nodes[tag_type] = opc_nodes.get(tag_type, [])
        id = make_opc_node_id(tag.name, ns)
        node = client.get_node(id)
        try:
            _ = node.read_browse_name()
        except BadNodeIdUnknown:
            # node does not exist in OPC server, create it
            # instantiate first action as random sample, store in OPC
            val = 0.0
            var_type = VariantType.Double
            if tag_type == 'action':
                val = initial_action[tag_idx]
            elif tag_type == 'observation':
                val = initial_observation[tag_idx]
            elif tag_type == 'meta':
                if tag.name == 'reward':
                    val = 0.0
                elif tag.name == 'truncated':
                    val = False
                    var_type = VariantType.Boolean
                elif tag.name == 'terminated':
                    val = False
                    var_type = VariantType.Boolean
            node = folder.add_variable(id, tag.name, val, var_type)
        opc_nodes[tag_type].append(node)

    # Run env forever using OPC for actions, observations, and rewards
    while True:
        # get the action values from OPC
        action_values = client.read_values(opc_nodes['action'])

        # read the observation from the environment, write to OPC
        observation, reward, terminated, truncated, info = env.step(action_values)
        if terminated or truncated:
            observation, info = env.reset()

        # write the observation values to OPC
        client.write_values(opc_nodes['observation'], observation.tolist())

        # write the reward to OPC
        client.write_values(opc_nodes['meta'], [reward, terminated, truncated])

        if sleep_sec:
            sleep(sleep_sec)


@load_config(Config, base='config/')
def main(cfg: Config):
    env: gym.Env = init_environment(cfg.env)
    _logger.info(f"Running OPC env simulation {env}")

    opc_url = cfg.env.opc_url

    # make OPC client (sync)
    client = Client(opc_url)
    try:
        client.connect()
        run(env, client, cfg)
    except Exception as e:
        _logger.exception(e)
    finally:
        client.disconnect()
        env.close()


if __name__ == "__main__":
    _logger = logging.getLogger(__name__)
    main()
