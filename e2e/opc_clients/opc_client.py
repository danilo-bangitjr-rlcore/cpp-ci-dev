import asyncio

from asyncua import Client, Node
from opc_node import ServerNode

import yaml
from pathlib import Path
import argparse
import sys
import logging
import math

logger = logging.getLogger(__name__)


async def create_node(
        tag_name: str, ns: int, root_node: Node, initial_val: float, dropout: bool = True
) -> ServerNode:
    server_node = ServerNode(
        tag_name=tag_name, namespace=ns, root_node=root_node, initial_val=initial_val, dropout=dropout
    )
    await server_node.initialize_node()
    return server_node


async def main(env_cfg_path):
    try:
        with open(env_cfg_path, "r") as f:
            env_cfg = yaml.safe_load(f)
    except OSError as e:
        print(e)
        print("""Make sure you are running opc_server.py in root of the git repo
                 i.e. python e2e/opc_server/opc_server.py
              """)
        sys.exit(1)

    url = env_cfg["opc"]["url"]
    namespace = env_cfg["opc"]["namespace"]

    ctl_tag_names = env_cfg["control_tags"]
    ctl_map = env_cfg["ctl_map"]
    setpoints = env_cfg["initial_setpoints"]
    obs_tag_names = env_cfg["obs_col_names"]

    dropout = env_cfg["opc"]["dropout"]

    all_server_nodes = {}

    print(f"Connecting to {url} ...")
    async with Client(url=url) as client:

        # Create node
        nsidx = await client.get_namespace_index(namespace)
        print(f"Namespace Index for '{namespace}': {nsidx}")
        rlcore_root = client.get_node("ns=2;i=1")

        for ctl_tag in ctl_tag_names:
            sp_tag = ctl_map[ctl_tag]
            val = setpoints[sp_tag]
            print(f"Creating variable {nsidx=} {ctl_tag=} {rlcore_root=} {val=}")
            all_server_nodes[ctl_tag] = await create_node(
                tag_name=ctl_tag, ns=nsidx, root_node=rlcore_root, initial_val=val, dropout=dropout
            )

        for sp_tag in setpoints:
            val = setpoints[sp_tag]
            print(f"Creating variable {nsidx=} {sp_tag=} {rlcore_root=} {val=}")
            all_server_nodes[sp_tag] = await create_node(
                tag_name=sp_tag, ns=nsidx, root_node=rlcore_root, initial_val=val, dropout=dropout
            )

        for obs_tag in obs_tag_names:
            val = 1.0
            print(f"Creating variable {nsidx=} {obs_tag=} {rlcore_root=} {val=}")
            all_server_nodes[obs_tag] = await create_node(
                tag_name=obs_tag, ns=nsidx, root_node=rlcore_root, initial_val=val, dropout=dropout
            )

        if "special_tags" in env_cfg:
            special_tag_names = env_cfg["special_tags"]
            for spc_tag in special_tag_names:
                if spc_tag in setpoints:
                    continue

                val = 0.0
                print(f"Creating variable {nsidx=} {spc_tag=} {rlcore_root=} {val=}")
                all_server_nodes[spc_tag] = await create_node(
                    tag_name=spc_tag, ns=nsidx, root_node=rlcore_root, initial_val=val, dropout=dropout
                )

        await asyncio.sleep(5)

        try:
            count = 0
            while True:
                await asyncio.sleep(1)
                for ctl_tag_name in ctl_tag_names:
                    # read ctl tag
                    ctl_server_node: ServerNode = all_server_nodes[ctl_tag_name]
                    assert ctl_server_node.node is not None
                    val = await ctl_server_node.node.read_value()
                    logger.warning(f"read val {val} from {ctl_tag_name}")

                    # write sp tag
                    sp_tag_name = ctl_map[ctl_tag_name]
                    sp_server_node: ServerNode = all_server_nodes[sp_tag_name]
                    assert sp_server_node.node is not None
                    logger.warning(f"writing val {val} from {sp_tag_name}")
                    await sp_server_node.node.write_value(val)

                for tag_name in obs_tag_names:
                    server_node: ServerNode = all_server_nodes[tag_name]
                    await server_node.step()

                    # put a sin wave in here so clients can see something is happening
                    if server_node.good_status:
                        assert server_node.node is not None
                        await server_node.node.write_value(math.sin(count / 10))

                count += 1

        except OSError as e:
            print(e)
            sys.exit(1)
        except KeyboardInterrupt:
            pass
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="env config file, usually in config/env/[your_env].yaml",
        default="config/env/opc_minimal_config.yaml"
    )
    args = parser.parse_args()
    env_cfg_path = Path(args.env)

    asyncio.run(main(env_cfg_path))
