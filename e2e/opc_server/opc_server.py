#!/usr/bin/env python

import asyncio
import math
import sys
from asyncua import Server, ua, Node
from asyncua.crypto.validator import CertificateValidator, CertificateValidatorOptions
import yaml
from asyncua.server.user_managers import CertificateUserManager
from asyncua.crypto.permission_rules import SimpleRoleRuleset
from asyncua.crypto.truststore import TrustStore
import logging
import random
from pathlib import Path
import argparse

from setup_e2e import setup
import tempfile


logger = logging.getLogger(__name__)
NS = 3
DROPOUT = True

SETPOINTS = {
    "AIC3730_SP": 9.7,  # pH
    "AIC3731_SP": 675.0,  # ORP
    "PDIC3738_SP": 2.4,  # DP
    "FIC3734_SP": 32.5,  # flow
}

CTL_MAP = {
    "ns=3;i=14": "AIC3730_SP",  # (pH)
    "ns=3;i=15": "AIC3731_SP",  # (ORP)
    "ns=3;i=17": "FIC3734_SP",  # (flow)
}


def get_nodeid(ns: int, tag_name: str) -> str:
    if ";" in tag_name:
        # tag name is already formatted
        # swap from i to s indentifier
        # tag_name = tag_name.replace("i=", "s=")
        return tag_name
    # tag name is not yet formatted
    return f"ns={ns};s={tag_name}"


class ServerNode:
    def __init__(
        self, tag_name: str, namespace: int, initial_val: float, root_node: Node
    ) -> None:
        self.namespace = namespace
        self.tag_name = tag_name
        self.initial_val = initial_val
        self.root_node = root_node
        self.good_status = True
        self.bad_counter = 0
        self.node: Node | None = None

    async def initialize_node(self) -> None:
        nodeid = get_nodeid(ns=self.namespace, tag_name=self.tag_name)
        logger.warning(f"Creating node {nodeid}")
        node = await self.root_node.add_variable(
            nodeid, self.tag_name, self.initial_val
        )
        await node.set_writable()
        self.node = node

    async def step(self) -> None:
        """
        Increments bad counter if status is bad.
        If bad counter exceeds 30, flips status to good
        """

        if self.good_status:
            go_bad = random.random() < 0.001 and DROPOUT
            if go_bad:
                await self.fail()

            return

        # status is bad
        self.bad_counter += 1
        if self.bad_counter >= 30:
            await self.recover()

    async def fail(self) -> None:
        assert self.node is not None
        self.good_status = False
        try:
            await self.node.delete()
            self.node = None
            logger.warning(f"Deleted node associated with tag {self.tag_name}")
        except Exception:
            logger.warning(f"Failed to delete node associated with tag {self.tag_name}")

    async def recover(self) -> None:
        # recreate deleted tag
        try:
            await self.initialize_node()
            self.good_status = True
            self.bad_counter = 0
        except Exception:
            logger.warning(
                f"Failed to create node associated with tag: {self.tag_name}"
            )


async def create_node(
    tag_name: str, ns: int, root_node: Node, initial_val: float
) -> ServerNode:
    server_node = ServerNode(
        tag_name=tag_name, namespace=ns, root_node=root_node, initial_val=initial_val
    )
    await server_node.initialize_node()
    return server_node


async def connect(cert_path: Path, env_cfg_path: Path):
    # certificate user manager associates public key certificates with
    # different users
    # the name param is actually not inportant since the cert is the identity here but should be unique
    #
    # currently you need to manually add these add_admin commands for each unique user connecting
    cert_user_manager = CertificateUserManager()
    # await cert_user_manager.add_admin(
    #     "certificates/trusted/certs/cert.pem",
    #     # "certificates/trusted/certs/admin.der",
    #     name="my_client"
    # )

    await cert_user_manager.add_admin(
        cert_path / "certificates/trusted/certs/telegraf.pem", name="telegraf-main"
    )
    await cert_user_manager.add_admin(
        cert_path / "certificates/trusted/certs/agent.pem", name="agent"
    )
    # await cert_user_manager.add_admin(
    #    "certificates/trusted/certs/test-client.der",
    #    name="test_client"
    # )

    server = Server(user_manager=cert_user_manager)
    await server.init()
    await server.set_application_uri("urn:rlcore-server")
    # bind to all local ip addresses on port 50005
    server.set_endpoint("opc.tcp://0.0.0.0:50005")

    # connecting clients must use Basic256Sha256 and SignAndEncrypt
    # SimpleRoleRuleset:
    # Admin -> read/write
    # User -> read only
    # Anonymous -> none
    server.set_security_policy(
        [ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt],
        permission_ruleset=SimpleRoleRuleset(),
    )

    # load the servers pub+private keys
    await server.load_certificate(str(cert_path / "certificates/server/cert.der"))
    await server.load_private_key(
        cert_path / "certificates/server/key.pem", password=None, format=".pem"
    )
    # this can be disabled for debugging
    # server.disable_clock(True)
    server.set_server_name("RLCore OPC-UA Server")

    # trusted client certificates MUST go in this directory, they won't be valid otherwise
    trust_store = TrustStore([Path("certificates/trusted/certs")], [])
    await trust_store.load()
    validator = CertificateValidator(
        options=CertificateValidatorOptions.TRUSTED_VALIDATION
        | CertificateValidatorOptions.PEER_CLIENT,
        trust_store=trust_store,
    )
    server.set_certificate_validator(validator=validator)

    uri = "urn:rlcore-server"
    idx = await server.register_namespace(uri)

    # this is the "root" node of the whole server
    objects = server.nodes.objects

    # this is the root node of our namespace
    rlcore_root = await objects.add_object(idx, "RLCore")

    # test_real = await rlcore_root.add_variable("ns=1;s=string.path",
    #                                            "TestReal",
    #                                            ua.Variant(256, ua.VariantType.Float))
    # await test_real.set_writable()

    # parse env config yaml file to create the tags that we need
    try:
        with open(env_cfg_path, "r") as f:
            env_cfg = yaml.safe_load(f)
    except OSError as e:
        print(e)
        print("""Make sure you are running opc_server.py in root of the git repo
                 i.e. python e2e/opc_server/opc_server.py
              """)
        sys.exit(1)

    all_server_nodes = {}

    logger.warning(env_cfg)
    obs_tag_names = env_cfg["obs_col_names"]
    ctl_tag_names = env_cfg["control_tags"]

    for ctl_tag in ctl_tag_names:
        sp_tag = CTL_MAP[ctl_tag]
        val = SETPOINTS[sp_tag]
        all_server_nodes[ctl_tag] = await create_node(
            tag_name=ctl_tag, ns=NS, root_node=rlcore_root, initial_val=val
        )

    for sp_tag in SETPOINTS:
        val = SETPOINTS[sp_tag]
        all_server_nodes[sp_tag] = await create_node(
            tag_name=sp_tag, ns=NS, root_node=rlcore_root, initial_val=val
        )

    for obs_tag in obs_tag_names:
        val = 1.0
        all_server_nodes[obs_tag] = await create_node(
            tag_name=obs_tag, ns=NS, root_node=rlcore_root, initial_val=val
        )

    if "special_tags" in env_cfg:
        special_tag_names = env_cfg["special_tags"]
        for spc_tag in special_tag_names:
            if spc_tag in SETPOINTS:
                continue

            val = 0.0
            all_server_nodes[spc_tag] = await create_node(
                tag_name=spc_tag, ns=NS, root_node=rlcore_root, initial_val=val
            )

    try:
        async with server:
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
                    sp_tag_name = CTL_MAP[ctl_tag_name]
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
        default="config/env/scrubber_online_test.yaml"
    )
    args = parser.parse_args()
    env_cfg_path = Path(args.env)

    file_path = Path(__file__).parent

    with tempfile.TemporaryDirectory(dir=file_path) as tmpdirname:
        cert_path = file_path / tmpdirname
        setup(cert_path, env_cfg_path=env_cfg_path)
        asyncio.run(connect(cert_path, env_cfg_path))
