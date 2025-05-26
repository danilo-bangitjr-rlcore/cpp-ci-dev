import logging
from datetime import UTC
from types import TracebackType
from typing import Optional, Type

import backoff
from asyncua import Client, Node, ua
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from pydantic import BaseModel, ConfigDict

from coreio.config import OPCConnectionConfig
from coreio.utils.io_events import OPCUANodeWriteValue
from corerl.corerl.data_pipeline.tag_config import TagConfig, TagType

logger = logging.getLogger(__name__)


class NodeData(BaseModel):
    node: Node
    var_type: ua.VariantType
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OPC_Connection:
    def __init__(self):
        self.opc_client: Client
        self.registered_nodes: dict[str, NodeData] = {}
        self._connected = False
        self.registered_nodes: dict[str, NodeData] = {}

    async def init(self, cfg: OPCConnectionConfig, tag_configs: list[TagConfig]):
        self.connection_id = cfg.connection_id
        self.opc_client = Client(cfg.opc_conn_url)
        await self._register_action_nodes(tag_configs)

        if cfg.client_cert_path and cfg.client_private_key_path and cfg.server_cert_path:
            assert cfg.application_uri is not None
            self.opc_client.application_uri = cfg.application_uri

            await self.opc_client.set_security(
                SecurityPolicyBasic256Sha256,
                certificate=cfg.client_cert_path,
                private_key=cfg.client_private_key_path,
                mode=ua.MessageSecurityMode.SignAndEncrypt,
                server_certificate=cfg.server_cert_path,
            )
        if cfg.client_cert_path or cfg.client_private_key_path or cfg.server_cert_path:
            logger.warning(
                f"OPC Clinent (connection_id: {cfg.connection_id}): " +
                "Client cert path, client private key path and server cert path " +
                "must be declared to set an encrypted connection.\n" +
                "Using default connection."
            )
        return self

    async def register_node(self, node_id: str):
        async with self.opc_client:
            if not node_id.startswith("ns="):
                raise ValueError(f"Problem encountered in tag config for {node_id} " +
                    "For ai_setpoint tags, node_identifier must be defined as the long-form OPC identifier")

            node = self.opc_client.get_node(node_id)
            var_type = await node.read_data_type_as_variant_type()
            logger.info(f"Registering heatbeat with OPC node id '{node_id}'")

            self.registered_nodes[node_id] = NodeData(node=node, var_type=var_type)

    async def _register_action_nodes(self, tag_configs: list[TagConfig]):
        """
        Register nodes that:
        1. Have the relevant connection_id
        2. Are ai_setpoints
        """

        async with self.opc_client:
            for tag_cfg in sorted(tag_configs, key=lambda cfg: cfg.name):

                if tag_cfg.connection_id != self.connection_id or tag_cfg.type != TagType.ai_setpoint:
                    continue

                if tag_cfg.node_identifier is not None and tag_cfg.node_identifier.startswith("ns="):
                    node_id = tag_cfg.node_identifier
                else:
                    raise ValueError(f"Problem encountered in tag config for {tag_cfg.name}: " +
                        "For ai_setpoint tags, node_identifier must be defined as the long-form OPC identifier")

                node = self.opc_client.get_node(node_id)
                var_type = await node.read_data_type_as_variant_type()
                logger.info(f"Registering node '{tag_cfg.name}' with OPC node id '{node_id}'")

                self.registered_nodes[node_id] = NodeData(node=node, var_type=var_type)


    @backoff.on_exception(backoff.expo, (OSError), max_time=30,)
    async def start(self):
        await self.opc_client.connect()
        return self

    async def cleanup(self):
        await self.opc_client.disconnect()
        return self

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ):
        _ = exc_type, exc, tb
        await self.cleanup()

    async def ensure_connected(self):
        try:
            await self.opc_client.check_connection()
        except ConnectionError:
            await self.opc_client.connect()

    @backoff.on_exception( backoff.expo, (ua.UaError, ConnectionError), max_time=30,)
    async def write_opcua_nodes(self, nodes_to_write: list[OPCUANodeWriteValue]):
        """
        Writing core-rl values into OPC
        Some checks might seem redundant with core-rl, but those will be removed from core-rl shortly
        """
        # Reconnect if connection is not ok
        await self.ensure_connected()

        nodes = []
        data_values = []

        for node in nodes_to_write:
            # Using get() instead of [], because it returns None instead of error if node_id is not found
            if node.node_id not in self.registered_nodes:
                logger.warning(f"Node {node.node_id} is unknown")
                continue

            node_entry = self.registered_nodes[node.node_id]

            var_type = node_entry.var_type
            if var_type in {
                ua.VariantType.SByte,
                ua.VariantType.Byte,
                ua.VariantType.Int16,
                ua.VariantType.UInt16,
                ua.VariantType.Int32,
                ua.VariantType.UInt32,
                ua.VariantType.Int64,
                ua.VariantType.UInt64,
            }:
                write_val = int(node.value)
            elif var_type in {ua.VariantType.Double, ua.VariantType.Float}:
                write_val = float(node.value)
            else:
                logger.warning(f"Var type of {var_type} is unknown in {node.node_id}")
                write_val = node.value

            # the source timestamp is sent to the OPC server, which itself has a server timestamp
            # recorded when it receives the write. if these values are too far apart, some OPC
            # implementations will consider the quality of this tag to be bad, so we need
            # to ensure that the values we write have an up-to-date timestamp
            # (and that they align with the server).
            dt = ua.uatypes.DateTime.now(UTC) # this is a load bearing timestamp
            data_value = ua.DataValue(ua.Variant(write_val, var_type), SourceTimestamp=dt)
            nodes.append(node_entry.node)
            data_values.append(data_value)

        if len(nodes) > 0:
            await self.opc_client.write_values(nodes, data_values)
