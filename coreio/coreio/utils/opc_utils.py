import logging

from asyncua.ua import VariantType
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import INTEGER, REAL

from coreio.communication.opc_communication import NodeData, OPC_Connection_IO
from coreio.config import OPCConnectionConfig, TagConfigAdapter
from coreio.utils.config_schemas import HeartbeatConfigAdapter

logger = logging.getLogger()


async def initialize_opc_connections(
        cfg_opc_connections: list[OPCConnectionConfig],
        cfg_tags: list[TagConfigAdapter],
        cfg_heartbeat: HeartbeatConfigAdapter | None = None,
) -> dict[str, OPC_Connection_IO]:

    opc_connections: dict[str, OPC_Connection_IO] = {}

    for opc_conn_cfg in cfg_opc_connections:
        logger.info(f"Connecting to OPC Connection {opc_conn_cfg.connection_id} at {opc_conn_cfg.opc_conn_url}")
        opc_conn = await OPC_Connection_IO().init(opc_conn_cfg)
        opc_connections[opc_conn_cfg.connection_id] = opc_conn

        logger.debug(f"Registering nodes for {opc_conn_cfg.connection_id}")
        await opc_conn.register_cfg_nodes(cfg_tags)

        if isinstance(cfg_heartbeat, HeartbeatConfigAdapter):
            logger.info("Only if data ingress is disabled, we need to register heartbeat separately."
                         "If data ingress is enabled, then it should be included in the data ingress tags",
                         )
            if cfg_heartbeat.connection_id == opc_conn_cfg.connection_id:
                heartbeat_id = cfg_heartbeat.heartbeat_node_id

                if heartbeat_id is not None:
                    logger.debug(f"Registering heartbeat for {cfg_heartbeat.connection_id}")
                    await opc_conn.register_node(heartbeat_id, "heartbeat")
                else:
                    logger.debug(f"No heartbeat found for {cfg_heartbeat.connection_id}")

    return opc_connections


def concat_opc_nodes(
        opc_connections: dict[str, OPC_Connection_IO],
        skip_heartbeat: bool = False,
        heartbeat_name: str = "heartbeat",
) -> dict[str, NodeData]:
    logger.debug("Concatenating all the opc nodes across all opc connections")
    all_registered_nodes: dict[str, NodeData] = {}
    for connection_id, opc_conn in opc_connections.items():
        logger.debug(f"Reading registered nodes from {connection_id}")
        for node_id, node in opc_conn.registered_nodes.items():
            if skip_heartbeat and node.name == heartbeat_name:
                continue
            if node_id in all_registered_nodes:
                logger.warning(f"Found repeat node_id of {node_id}, overwriting with {connection_id} {node_id}")
            all_registered_nodes[node_id] = node
    return all_registered_nodes

OPC_TO_SQLALCHEMY_TYPE_MAP = {
    # Null and Basic Types
    VariantType.Null: String(),  # Store as nullable string or could use JSON
    VariantType.Boolean: Boolean(),

    # Integer Types
    VariantType.SByte: INTEGER(),
    VariantType.Byte: INTEGER(),
    VariantType.Int16: INTEGER(),
    VariantType.UInt16: INTEGER(),
    VariantType.Int32: INTEGER(),
    VariantType.UInt32: INTEGER(),
    VariantType.Int64: INTEGER(),
    VariantType.UInt64: INTEGER(),

    # Floating Point Types
    VariantType.Float: REAL(),
    VariantType.Double: REAL(),

    # String and Data Types
    VariantType.String: Text(),
    VariantType.DateTime: DateTime(timezone=True), # Configurable timezone?
    VariantType.Guid: String(36),
    VariantType.ByteString: LargeBinary(),
    VariantType.XmlElement: Text(),

    # OPC Specific Types (stored as strings/JSON for flexibility)
    VariantType.NodeId: String(255),
    VariantType.ExpandedNodeId: String(512),
    VariantType.StatusCode: Integer(),
    VariantType.QualifiedName: String(512),
    VariantType.LocalizedText: JSON(),

    # Complex Types
    VariantType.ExtensionObject: JSON(),
    VariantType.DataValue: JSON(),
    VariantType.Variant: JSON(),
    VariantType.DiagnosticInfo: JSON(),
}
