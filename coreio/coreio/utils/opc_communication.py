import logging
from collections.abc import Sequence
from datetime import UTC
from types import TracebackType
from typing import Any, Protocol, assert_never

import backoff
from asyncua import Client, Node, ua
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from corerl.data_pipeline.tag_config import TagType
from pydantic import BaseModel, ConfigDict

from coreio.config import (
    OPCAuthMode,
    OPCAuthModeConfig,
    OPCAuthModeUsernamePasswordConfig,
    OPCConnectionConfig,
    OPCMessageSecurityMode,
    OPCSecurityPolicyBasic256SHA256Config,
    OPCSecurityPolicyConfig,
    OPCSecurityPolicyNoneConfig,
)
from coreio.utils.io_events import OPCUANodeWriteValue

logger = logging.getLogger(__name__)


class NodeData(BaseModel):
    node: Node
    var_type: ua.VariantType
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TagConfig(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def connection_id(self) -> str | None: ...
    @property
    def node_identifier(self) -> str | None: ...
    @property
    def type(self) -> TagType: ...


def log_backoff(details: Any):
    wait = details["wait"]
    tries = details["tries"]
    func = details["target"].__name__
    logger.warning(f"Backing off {wait:.1f} seconds after {tries} tries calling {func}")

class OPC_Connection:
    def __init__(self):
        self.opc_client: Client | None = None
        self.registered_nodes: dict[str, NodeData] = {}

    async def init(self, cfg: OPCConnectionConfig, tag_configs: Sequence[TagConfig]):
        self.connection_id = cfg.connection_id
        self.opc_client = Client(cfg.opc_conn_url)
        self._connected = False

        assert cfg.application_uri is not None
        self.opc_client.application_uri = cfg.application_uri

        await self._set_security_policy(cfg.security_policy)
        await self._set_auth_mode(cfg.authentication_mode)

        await self._register_action_nodes(tag_configs)

        return self


    @backoff.on_exception(backoff.expo, Exception, max_value=30, on_backoff=log_backoff)
    async def ensure_connected(self):
        assert self.opc_client is not None, 'OPC client is not initialized'

        if self._connected is False:
            await self.opc_client.connect()
            self._connected = True

        try:
            await self.opc_client.check_connection()

        except Exception:
            await self.opc_client.connect()
            self._connected = True
            await self.opc_client.check_connection()

        return self.opc_client


    async def _set_security_policy(self, policy: OPCSecurityPolicyConfig):
        assert self.opc_client is not None

        match policy:
            case OPCSecurityPolicyBasic256SHA256Config():
                mode = (
                    ua.MessageSecurityMode.Sign
                    if policy.mode is OPCMessageSecurityMode.sign
                    else ua.MessageSecurityMode.SignAndEncrypt
                )
                await self.opc_client.set_security(
                    SecurityPolicyBasic256Sha256,
                    certificate=policy.client_cert_path,
                    private_key=policy.client_key_path,
                    mode=mode,
                    server_certificate=str(policy.server_cert_path),
                )

            case OPCSecurityPolicyNoneConfig():
                pass

            case _:
                assert_never(policy)

    async def _set_auth_mode(self, auth_mode: OPCAuthModeConfig):
        assert self.opc_client is not None

        match auth_mode:
            case OPCAuthModeUsernamePasswordConfig():
                self.opc_client.set_user(auth_mode.username)
                self.opc_client.set_password(auth_mode.password)

            case OPCAuthMode.anonymous:
                pass

            case _:
                assert_never(auth_mode)

    async def register_node(self, node_id: str):
        client = await self.ensure_connected()

        if not node_id.startswith("ns="):
            raise ValueError(f"Problem encountered in tag config for {node_id} " +
                "For ai_setpoint tags, node_identifier must be defined as the long-form OPC identifier")

        node = client.get_node(node_id)
        var_type = await node.read_data_type_as_variant_type()
        logger.info(f"Registering heatbeat with OPC node id '{node_id}'")

        self.registered_nodes[node_id] = NodeData(node=node, var_type=var_type)

    async def _register_action_nodes(self, tag_configs: Sequence[TagConfig]):
        """
        Register nodes that:
        1. Have the relevant connection_id
        2. Are ai_setpoints
        """
        for tag_cfg in tag_configs:
            if tag_cfg.connection_id != self.connection_id or tag_cfg.type != TagType.ai_setpoint:
                continue

            if tag_cfg.node_identifier is None:
                raise ValueError(f"Problem encountered in tag config for {tag_cfg.name}: " +
                    "For ai_setpoint tags, node_identifier must be defined")

            await self.register_node(tag_cfg.node_identifier)


    @backoff.on_exception(backoff.expo, Exception, max_value=30, on_backoff=log_backoff)
    async def start(self):
        await self.ensure_connected()
        return self

    async def cleanup(self):
        if self.opc_client is None:
            return self

        await self.opc_client.disconnect()
        self._connected = False
        return self

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ):
        _ = exc_type, exc, tb
        await self.cleanup()

    @backoff.on_exception(backoff.expo, (ua.UaError, ConnectionError), max_value=30, on_backoff=log_backoff)
    async def write_opcua_nodes(self, nodes_to_write: list[OPCUANodeWriteValue]):
        assert self.opc_client is not None, 'OPC client is not initialized'
        # Reconnect if connection is not ok
        await self.ensure_connected()

        nodes = []
        data_values = []

        for node in nodes_to_write:
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
