# instantiate our OPC client
# inttantiate a farama gym environment instance
# all gym environment actions have side effects that interact with OPC client
# all agent observations are performed through timescale DB

import logging
from datetime import timedelta

import gymnasium as gym
import numpy as np
from asyncua.sync import Client, SyncNode
from asyncua.ua.uaerrors import BadNodeIdExists, BadNodeIdUnknown
from asyncua.ua.uatypes import VariantType

from corerl.configs.config import MISSING, config
from corerl.configs.loader import load_config
from corerl.environment.async_env.async_env import GymEnvConfig
from corerl.environment.factory import init_environment
from corerl.utils.opc_connection import make_opc_node_id
from corerl.utils.time import clock_generator, wait_for_timestamp


@config()
class OPCSimConfig:
    gym: GymEnvConfig = MISSING
    sim_timestep_period: timedelta | None = None
    obs_tags: list[str] = MISSING
    action_tags: list[str] = MISSING
    opc_server: str = "0.0.0.0"

def initialize_opc_folder(client: Client, cfg_env: GymEnvConfig):
    # create folder containing environment variables
    folder_node_id = make_opc_node_id(cfg_env.gym_name)
    try:
        folder = client.nodes.objects.add_folder(folder_node_id, cfg_env.gym_name)
    except BadNodeIdExists:
        # folder already exists
        folder = client.get_node(folder_node_id)
    return folder


def initialize_opc_nodes_from_tags(
    client: Client,
    cfg: OPCSimConfig,
    initial_observation: np.ndarray,
    initial_action: np.ndarray,
):
    folder = initialize_opc_folder(client, cfg.gym)
    # create OPC nodes based on tags
    opc_nodes: dict[str, list[SyncNode]] = {}

    obs_nodes = get_nodes(tags=cfg.obs_tags, initial_vals=initial_observation, folder=folder, client=client)
    action_nodes = get_nodes(tags=cfg.action_tags, initial_vals=initial_action, folder=folder, client=client)
    heartbeat_node = get_nodes(tags=["heartbeat"], initial_vals=np.array([0]), folder=folder, client=client)
    agent_step_node = get_nodes(tags=["agent_step"], initial_vals=np.array([0]), folder=folder, client=client)

    opc_nodes["observation"] = obs_nodes
    opc_nodes["action"] = action_nodes
    opc_nodes["heartbeat"] = heartbeat_node
    opc_nodes["agent_step"] = agent_step_node
    return opc_nodes


def get_nodes(tags: list[str], initial_vals: np.ndarray, folder: SyncNode, client: Client) -> list[SyncNode]:
    nodes = []
    for idx, tag in enumerate(tags):
        id = make_opc_node_id(tag, 2)
        node = client.get_node(id)

        try:
            _ = node.read_browse_name()
        except BadNodeIdUnknown:
            # node does not exist in OPC server, create it
            # instantiate first action as random sample, store in OPC
            val = 0.0
            var_type = VariantType.Double
            val = initial_vals[idx]
            node = folder.add_variable(id, tag, val, var_type)

        nodes.append(node)

    return nodes

def run(env: gym.Env, client: Client, cfg: OPCSimConfig):
    seed = cfg.gym.seed

    initial_observation, info = env.reset(seed=seed)
    initial_action = env.action_space.sample()

    opc_nodes = initialize_opc_nodes_from_tags(
        client, cfg, initial_observation, initial_action
    )

    # Run simulation forever using OPC for communication
    if cfg.sim_timestep_period is not None:
        sync = wait_for_sim_step(cfg.sim_timestep_period)
    else:
        sync = wait_for_agent_step(client, opc_nodes["agent_step"][0])

    while True:
        next(sync) # waits to sync with agent step or sim step
        _logger.info("OPC Sim Step")

        # get the action values from OPC
        action_values = client.read_values(opc_nodes["action"])

        # read the observation from the environment, write to OPC
        observation, reward, terminated, truncated, info = env.step(action_values)
        if terminated or truncated:
            observation, info = env.reset()

        # write the observation values to OPC
        client.write_values(opc_nodes["observation"], observation.tolist())


def wait_for_sim_step(timestep_period: timedelta):
    """
    Using a generator here keeps the main control loop logic cleaner.
    The last sim timestamp is tracked within the generator and will persist
    between calls of "next"
    """
    timestep_clock = clock_generator(tick_period=timestep_period)
    while True:
        next_sim_timestep = next(timestep_clock)
        wait_for_timestamp(next_sim_timestep)
        yield


def wait_for_agent_step(client: Client, agent_step_node: SyncNode):
    """
    Using a generator here keeps the main control loop logic cleaner.
    The last agent step is tracked within the generator and will persist
    between calls of "next"
    """
    last_agent_step = client.read_values([agent_step_node])[0]
    while True:
        agent_step = client.read_values([agent_step_node])[0]
        if agent_step != last_agent_step:
            last_agent_step = agent_step
            yield

@load_config(OPCSimConfig, base="config/")
def main(cfg: OPCSimConfig):
    env: gym.Env = init_environment(cfg.gym)
    _logger.info(f"Running OPC env simulation {env}")

    client = Client(f"opc.tcp://admin@{cfg.opc_server}:4840/rlcore/server/")
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
    log_fmt = "[%(asctime)s][%(levelname)s] - %(message)s"
    logging.basicConfig(format=log_fmt, encoding="utf-8", level=logging.INFO)
    logging.getLogger('asyncua').setLevel(logging.CRITICAL)
    main()
