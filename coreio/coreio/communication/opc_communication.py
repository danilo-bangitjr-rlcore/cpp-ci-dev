from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any, Concatenate, assert_never, cast

import backoff
from asyncua import Client, ua
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from asyncua.ua.uaerrors import BadNodeIdUnknown
from cryptography import x509
from cryptography.x509.oid import ExtensionOID
from lib_utils.opc.opc_communication import NodeData, OPC_Connection, log_backoff

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

class OPC_Connection_IO(OPC_Connection):
    def __init__(self):
        super().__init__()

    # -------------------- #
    # --- Init methods --- #
    # -------------------- #

    async def init(self, cfg: OPCConnectionConfig):
        self.connection_id = cfg.connection_id
        logger.debug(f"Starting initialization for {self.connection_id}")
        self.opc_client = Client(cfg.opc_conn_url, timeout=30, watchdog_intervall=30)
        self._connected = False

        self._set_application_uri(cfg)

        logger.debug(f"Setting security policy for {self.connection_id}")
        await self._set_security_policy(cfg.security_policy)
        logger.debug(f"Setting auth mode for {self.connection_id}")
        await self._set_auth_mode(cfg.authentication_mode)

        # Test connection
        logger.debug(f"Ensuring connection for {self.connection_id}")
        await self.ensure_connected()

        return self

    def _set_application_uri(self, cfg: OPCConnectionConfig):
        logger.info(f"Setting application uri for {self.connection_id}")
        assert self.opc_client is not None, "OPC client is not initialized"
        if cfg.application_uri is not None:
            self.opc_client.application_uri = cfg.application_uri
            return

        logger.info("No uri found in cfg, attempting to read from cert")
        if isinstance(cfg.security_policy, OPCSecurityPolicyBasic256SHA256Config):
            client_uri = get_application_uri_from_cert(cfg.security_policy.client_cert_path)
            if client_uri is not None:
                self.opc_client.application_uri = client_uri
                return

        logger.error(f"Client URI not found for {self.connection_id}... Continuing execution")

    async def _set_security_policy(self, policy: OPCSecurityPolicyConfig):
        assert self.opc_client is not None

        logger.debug(f"Security policy for {self.connection_id} is {policy}" )
        match policy:
            case OPCSecurityPolicyBasic256SHA256Config():
                mode = (
                    ua.MessageSecurityMode.Sign
                    if policy.mode is OPCMessageSecurityMode.sign
                    else ua.MessageSecurityMode.SignAndEncrypt
                )
                logger.debug(f"Setting security in opc client"
                             f" client cert path {policy.client_cert_path}, client key path {policy.client_key_path},"
                             f" server cert path {policy.server_cert_path}")

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
        logger.debug(f"Auth mode for {self.connection_id} is {auth_mode}")

        match auth_mode:
            case OPCAuthModeUsernamePasswordConfig():
                logger.debug(f"Setting user and password in opc client for {self.connection_id}")
                self.opc_client.set_user(auth_mode.username)
                self.opc_client.set_password(auth_mode.password)

            case OPCAuthMode.anonymous:
                pass

            case _:
                assert_never(auth_mode)

    # ------------------------- #
    # --- Connection Health --- #
    # ------------------------- #

    @staticmethod
    def ensure_healthy_connection[**P, R](
        func: Callable[Concatenate[OPC_Connection_IO, P], Awaitable[R]],
    ) -> Callable[Concatenate[OPC_Connection_IO, P], Awaitable[R]]:
        async def wrapper(self: OPC_Connection_IO, *args: P.args, **kwargs: P.kwargs) -> R:
            try:
                logger.debug(f"Checking connection for {self.connection_id} from wrapper")
                await self.ensure_connected()
            except Exception as exc:
                logger.warning(f"Failed to establish healthy connection for {self.connection_id}: {exc}")
                self._connected = False
                raise

            return await func(self, *args, **kwargs)
        return wrapper

    @backoff.on_exception(backoff.expo, Exception, max_value=MAX_BACKOFF_SECONDS, on_backoff=log_backoff)
    async def ensure_connected(self):
        try:
            logger.debug(f"About to establish persistent connection to {self.connection_id}")
            await self.ensure_connected_no_backoff()
            logger.info(f"Successfully established persistent connection to {self.connection_id}")

        except Exception as exc:
            logger.error(f"Failed to establish persistent connection to {self.connection_id}: {exc}")
            raise

    async def cleanup(self) -> OPC_Connection_IO:
        if self.opc_client is None:
            logger.debug(f"OPC client for {self.connection_id} is already None")
            return self

        try:
            await self.opc_client.disconnect()
            self._connected = False
            logger.info(f"Successfully cleaned up connection {self.connection_id}")

        except Exception as exc:
            logger.error(f"Error cleaning up connection {self.connection_id}: {exc}")

        return self

    # ------------------ #
    # --- IO Methods --- #
    # ------------------ #

    @ensure_healthy_connection
    @backoff.on_exception(backoff.expo, BadNodeIdUnknown, max_value=MAX_BACKOFF_SECONDS, on_backoff=log_backoff)
    async def register_node(self, node_id: str, name: str):
        assert self.opc_client is not None, 'OPC client is not initialized'
        if not node_id.startswith("ns="):
            raise ValueError(f"Problem encountered in tag config for {node_id} " +
                "For ai_setpoint tags, node_identifier must be defined as the long-form OPC identifier")

        logger.info(f"Registering OPC node with id '{node_id}'")
        node = self.opc_client.get_node(node_id)
        var_type = await node.read_data_type_as_variant_type()
        logger.debug(f"Variant type for {node_id} is {var_type}")

        self.registered_nodes[node_id] = NodeData(node=node, var_type=var_type, name=name)

    @ensure_healthy_connection
    async def register_cfg_nodes(self, tag_configs: Sequence[TagConfigAdapter]):
        assert self.opc_client is not None, 'OPC client is not initialized'
        for tag_cfg in tag_configs:
            if tag_cfg.connection_id != self.connection_id or tag_cfg.node_identifier is None:
                continue

            await self.register_node(tag_cfg.node_identifier, tag_cfg.name)


    @ensure_healthy_connection
    @backoff.on_exception(backoff.expo, BadNodeIdUnknown, max_value=MAX_BACKOFF_SECONDS, on_backoff=log_backoff)
    async def write_opcua_nodes(self, nodes_to_write: Sequence[OPCUANodeWriteValue]):
        assert self.opc_client is not None, 'OPC client is not initialized'
        assert (
            {node.node_id for node in nodes_to_write} <= self.registered_nodes.keys()
        ), "Not all nodes_to_write are in our registered_nodes"
        nodes = []
        data_values = []

        for node in nodes_to_write:
            logger.debug(f"Parsing {node.node_id} in {self.connection_id}, about to write...")
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

                logger.debug(f"Added to write list {node.node_id} in {self.connection_id}, about to write...")
                nodes.append(node_entry.node)
                data_values.append(data_value)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to convert value {node.value} to {var_type} for node {node.node_id}: {e}")

        if len(nodes) > 0:
            logger.debug(f"About to write values in {self.connection_id}")
            await self.opc_client.write_values(nodes, data_values)

    @ensure_healthy_connection
    async def _read_opcua_nodes(self, nodes_to_read: dict[str, NodeData]):
        assert self.opc_client is not None, 'OPC client is not initialized'
        assert nodes_to_read.keys() <= self.registered_nodes.keys(), "Not all nodes_to_read are in our registered_nodes"
        opc_nodes_to_read = [node.node for node in nodes_to_read.values()]
        read_values = [None] * len(opc_nodes_to_read)
        try:
            logger.debug(f"Reading values in {self.connection_id}")
            read_values = await self.opc_client.read_values(opc_nodes_to_read)
        except Exception as exc:
            logger.error(f"Error on bulk read, returning [Nones]:\n{exc!s}")

        return read_values

    async def read_nodes_named(self, nodes_to_read: dict[str, NodeData]) -> dict[str, Any]:
        logger.debug(f"About to read {len(nodes_to_read)} named nodes from {self.connection_id}")
        read_values = await self._read_opcua_nodes(nodes_to_read)

        nodes_name_val = {}
        for node, read_value in zip(nodes_to_read.values(), read_values, strict=True):
            nodes_name_val[node.name] = read_value

        return nodes_name_val

def get_application_uri_from_cert(cert_path: Path):
    """Extract application URI from certificate's SAN extension"""
    # Load certificate
    with open(cert_path, 'rb') as f:
        cert_data = f.read()

    # Handle both PEM and DER formats
    try:
        cert = x509.load_pem_x509_certificate(cert_data)
    except ValueError:
        cert = x509.load_der_x509_certificate(cert_data)

    try:
        # Get Subject Alternative Name extension
        san = cast(
            x509.SubjectAlternativeName,
            cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME).value,
        )

        # Extract URI values
        uris = san.get_values_for_type(x509.UniformResourceIdentifier)

        # Return first URI (typically the application URI)
        if uris:
            return uris[0]
        return None

    except x509.ExtensionNotFound:
        return None
