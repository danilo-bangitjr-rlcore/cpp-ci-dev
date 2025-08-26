from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import UTC
from types import TracebackType
from typing import Any, assert_never

import backoff
from asyncua import Client, Node, ua
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from asyncua.ua.uaerrors import BadNodeIdUnknown
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
    TagConfigAdapter,
)
from coreio.utils.io_events import OPCUANodeWriteValue

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 30

class NodeData(BaseModel):
    node: Node
    var_type: ua.VariantType
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

def log_backoff(details: Any):
    wait = details["wait"]
    tries = details["tries"]
    func = details["target"].__name__
    logger.error(f"Backing off {wait:.1f} seconds after {tries} tries calling {func}")

class OPC_Connection:
    def __init__(self):
        self.opc_client: Client | None = None
        self.registered_nodes: dict[str, NodeData] = {}
        self._context_active = False

    # -------------------- #
    # --- Init methods --- #
    # -------------------- #

    async def init(self, cfg: OPCConnectionConfig):
        self.connection_id = cfg.connection_id
        self.opc_client = Client(cfg.opc_conn_url)
        self._connected = False

        self.opc_client.application_uri = cfg.application_uri

        await self._set_security_policy(cfg.security_policy)
        await self._set_auth_mode(cfg.authentication_mode)

        # Test connection
        await self.ensure_connected()

        return self

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


    # -------------------------- #
    # --- Manage Connections --- #
    # -------------------------- #

    @backoff.on_exception(backoff.expo, Exception, max_value=MAX_BACKOFF_SECONDS, on_backoff=log_backoff)
    async def ensure_connected(self):
        assert self.opc_client is not None, 'OPC client is not initialized'

        if self._connected is False:
            await self.opc_client.connect()
            self._connected = True

        try:
            await self.opc_client.check_connection()

        except Exception:
            await self.opc_client.connect()
            await self.opc_client.check_connection()
            self._connected = True
            logger.error(f"Problem connecting to OPC server in {self.connection_id}")

        return self.opc_client

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
        self._context_active = True
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ):
        _ = exc_type, exc, tb
        self._context_active = False
        await self.cleanup()

    @staticmethod
    def requires_context(func: Callable[..., Any]):
        """Decorator that ensures method is called within active context"""
        async def wrapper(self: OPC_Connection, *args: Any, **kwargs: Any):
            if not self._context_active:
                raise RuntimeError(f"Function {func.__name__} must be called within the OPC context manager")
            return await func(self, *args, **kwargs)
        return wrapper

    # ------------------ #
    # --- IO Methods --- #
    # ------------------ #
    # All of these use @requires_context

    @requires_context
    @backoff.on_exception(backoff.expo, BadNodeIdUnknown, max_value=MAX_BACKOFF_SECONDS, on_backoff=log_backoff)
    async def register_node(self, node_id: str, name: str):
        assert self.opc_client is not None, 'OPC client is not initialized'
        if not node_id.startswith("ns="):
            raise ValueError(f"Problem encountered in tag config for {node_id} " +
                "For ai_setpoint tags, node_identifier must be defined as the long-form OPC identifier")

        logger.info(f"Registering OPC node with id '{node_id}'")
        node = self.opc_client.get_node(node_id)
        var_type = await node.read_data_type_as_variant_type()

        self.registered_nodes[node_id] = NodeData(node=node, var_type=var_type, name=name)

    @requires_context
    async def register_cfg_nodes(self, tag_configs: Sequence[TagConfigAdapter], ai_setpoint_only: bool = False):
        assert self.opc_client is not None, 'OPC client is not initialized'
        for tag_cfg in tag_configs:
            if tag_cfg.connection_id != self.connection_id or tag_cfg.node_identifier is None:
                continue

            await self.register_node(tag_cfg.node_identifier, tag_cfg.name)


    @requires_context
    @backoff.on_exception(backoff.expo, BadNodeIdUnknown, max_value=MAX_BACKOFF_SECONDS, on_backoff=log_backoff)
    async def write_opcua_nodes(self, nodes_to_write: Sequence[OPCUANodeWriteValue]):
        assert self.opc_client is not None, 'OPC client is not initialized'
        assert (
            {node.node_id for node in nodes_to_write} <= self.registered_nodes.keys()
        ), "Not all nodes_to_write are in our registered_nodes"
        nodes = []
        data_values = []

        for node in nodes_to_write:
            if node.node_id not in self.registered_nodes:
                logger.warning(f"Node {node.node_id} is unknown")
                continue

            node_entry = self.registered_nodes[node.node_id]

            var_type = node_entry.var_type

            try:
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
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to convert value {node.value} to {var_type} for node {node.node_id}: {e}")

        if len(nodes) > 0:
            await self.opc_client.write_values(nodes, data_values)

    @requires_context
    @backoff.on_exception(backoff.expo, BadNodeIdUnknown, max_value=MAX_BACKOFF_SECONDS, on_backoff=log_backoff)
    async def _read_opcua_nodes(self, nodes_to_read: dict[str, NodeData]):
        assert self.opc_client is not None, 'OPC client is not initialized'
        assert nodes_to_read.keys() <= self.registered_nodes.keys(), "Not all nodes_to_read are in our registered_nodes"
        opc_nodes_to_read = [node.node for node in nodes_to_read.values()]
        return await self.opc_client.read_values(opc_nodes_to_read)

    @requires_context
    async def read_nodes_named(self, nodes_to_read: dict[str, NodeData]) -> dict[str, Any]:
        read_values = await self._read_opcua_nodes(nodes_to_read)

        nodes_name_val = {}
        for node, read_value in zip(nodes_to_read.values(), read_values, strict=True):
            nodes_name_val[node.name] = read_value

        return nodes_name_val
