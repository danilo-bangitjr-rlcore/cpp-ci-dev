from datetime import UTC
import logging
from typing import Optional, Type
from asyncua import Client, Node, ua

from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from pydantic import BaseModel, ConfigDict

from coreio.config import CoreIOConfig

from corerl.data_pipeline.tag_config import TagConfig, TagType

logger = logging.getLogger(__name__)

# See: from corerl.environment.async_env.deployment_async_env import ActionNodeData
# class ActionNodeData(TypedDict):
#     connection_id: str
#     node_id: str
#     data_type: OPCUADataType

class NodeData(BaseModel):
    node: Node
    node_id: str
    var_type: ua.VariantType
    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: None | float

class OPC_Communication:
    def __init__(self):
        self.opc_client: Client 
        self.action_nodes: dict[str, NodeData] = {}
        self._connected = False

    async def init(self, cfg: CoreIOConfig, tag_configs: list[TagConfig]):
        self.opc_client = Client(cfg.opc_conn_url)

        self.action_nodes = await self._register_actions(self.opc_client, tag_configs)

        if cfg.client_cert_path and cfg.client_private_key_path:
            assert cfg.application_uri is not None
            self.opc_client.application_uri = cfg.application_uri

            await self.opc_client.set_security(
                SecurityPolicyBasic256Sha256,
                certificate=cfg.client_cert_path,
                private_key=cfg.client_private_key_path,
                mode=ua.MessageSecurityMode.SignAndEncrypt,
                server_certificate=cfg.server_cert_path,
            )
        return self

    async def _register_actions(self, opc_client: Client, tag_configs: list[TagConfig]):
        action_nodes: dict[str, NodeData] = {}

        async with opc_client:
            for tag_cfg in sorted(tag_configs, key=lambda cfg: cfg.name):
                if tag_cfg.type != TagType.ai_setpoint:
                    continue

                tag_name = tag_cfg.name

                if tag_cfg.node_identifier is not None and tag_cfg.node_identifier.startswith("ns="):
                    id = tag_cfg.node_identifier
                else:
                    raise ValueError(f"Problem encountered in tag config for {tag_cfg.name}: " +
                        "For ai_setpoint tags, node_identifier must be defined as the long-form OPC identifier")

                node = opc_client.get_node(id)
                var_type = await node.read_data_type_as_variant_type()
                logger.info(f"Registering action '{tag_name}' with OPC node id '{id}'")

                action_nodes[tag_name] = NodeData(node=node, node_id=id, var_type=var_type, value=None)

        return action_nodes

    async def __aenter__(self):
        await self.opc_client.connect()
        self._connected = True
        return self

    async def __aexit__(
        self,
        _exec_type : Optional[Type[BaseException]],
        _exec_val : Optional[BaseException],
        _exec_tb : Optional[object]
    ):
        _ = _exec_type, _exec_val, _exec_tb # Stop LSP from complaining about unused variables
        await self.opc_client.disconnect()
        return self

    async def emit_action(self, action_nodes: dict[str, NodeData]):
        if not self._connected:
            raise RuntimeError("OPC Client is not connected")

        nodes = []
        data_values = []

        for action_name in action_nodes.keys():
            node = action_nodes[action_name].node
            var_type = action_nodes[action_name].var_type
            action_val = action_nodes[action_name].value
            # the source timestamp is sent to the OPC server, which itself has a server timestamp
            # recorded when it receives the write. if these values are too far apart, some OPC
            # implementations will consider the quality of this tag to be bad, so we need
            # to ensure that the values we write have an up-to-date timestamp
            # (and that they align with the server).
            dt = ua.uatypes.DateTime.now(UTC) # this is a load bearing timestamp
            data_value = ua.DataValue(ua.Variant(action_val, var_type), SourceTimestamp=dt)
            nodes.append(node)
            data_values.append(data_value)

        await self.opc_client.write_values(nodes, data_values)
