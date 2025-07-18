import logging

from asyncua.ua import VariantType
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Double,
    Float,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
)

from coreio.communication.opc_communication import NodeMap, NodeRegistry, OPC_Connection

logger = logging.getLogger()

def concat_opc_nodes(
        opc_connections: dict[str, OPC_Connection],
        skip_heartbeat: bool = False,
        heartbeat_name: str = "heartbeat",
) -> NodeRegistry:
    all_registered_nodes: NodeRegistry = {}
    for connection_id, opc_conn in opc_connections.items():
        all_registered_nodes[connection_id] = {}
        for node_id, node in opc_conn.registered_nodes.items():
            if node_id in all_registered_nodes:
                logger.warning(
                    f"Node id {node_id} in OPC connection {connection_id} is not unique. "
                    "Details will be overwritten.",
                )
            if skip_heartbeat and node.name == heartbeat_name:
                continue
            all_registered_nodes[connection_id][node_id] = node
    return all_registered_nodes

def flatten_node_registry(node_registry: NodeRegistry) -> NodeMap:
    ...

OPC_TO_SQLALCHEMY_TYPE_MAP = {
    # Null and Basic Types
    VariantType.Null: String(),  # Store as nullable string or could use JSON
    VariantType.Boolean: Boolean(),

    # Integer Types
    VariantType.SByte: SmallInteger(),
    VariantType.Byte: SmallInteger(),
    VariantType.Int16: SmallInteger(),
    VariantType.UInt16: Integer(),
    VariantType.Int32: Integer(),
    VariantType.UInt32: BigInteger(),
    VariantType.Int64: BigInteger(),
    VariantType.UInt64: BigInteger(),

    # Floating Point Types
    VariantType.Float: Float(),
    VariantType.Double: Double(),

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
