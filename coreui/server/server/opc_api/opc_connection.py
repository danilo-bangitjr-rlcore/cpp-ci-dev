import logging

from asyncua import Client, Node
from asyncua.ua import NodeClass, NodeId
from lib_utils.opc.opc_communication import OPC_Connection

from server.opc_api.opc_models import NodeDetails, NodeInfo, ServerInfo, StatusResponse

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 30


def get_node_class_name(value: int) -> str:
    """Get the name of the NodeClass from its integer value."""
    try:
        return NodeClass(value).name
    except ValueError:
        return "Unknown"


class OPC_Connection_UI(OPC_Connection):
    def __init__(self):
        super().__init__()
        self.server_url: str = ""
        self._connected = False

    def _format_node_id(self, nodeid: NodeId) -> str:
        """Format NodeId object to standard OPC UA format (ns=X;i=Y)"""
        try:
            if hasattr(nodeid, 'NamespaceIndex') and hasattr(nodeid, 'Identifier'):
                return f"ns={nodeid.NamespaceIndex};i={nodeid.Identifier}"
            return str(nodeid)
        except Exception:
            return str(nodeid)

    async def _cast_array_value(self, value: str, dt_display_name: str) -> list:
        """Cast a string value to an array of the appropriate type"""
        if dt_display_name == "String":
            return [str(value)]
        # For other types, try to parse as comma-separated values
        try:
            if dt_display_name in ["Float", "Double"]:
                return [float(v.strip()) for v in value.split(',')]
            if dt_display_name in ["Int16", "Int32", "Int64", "UInt16", "UInt32", "UInt64"]:
                return [int(v.strip()) for v in value.split(',')]
            return [str(value)]
        except ValueError:
            return [str(value)]

    async def _cast_single_value(self, value: str, dt_display_name: str):
        """Cast a string value to the appropriate single type"""
        if dt_display_name in ["Float", "Double"]:
            return float(value)
        if dt_display_name in ["Int16", "Int32", "Int64", "UInt16", "UInt32", "UInt64"]:
            return int(value)
        if dt_display_name == "Boolean":
            return value.lower() in ("true", "1", "yes", "on")
        if dt_display_name == "String":
            return str(value)
        # For unknown types, try to write as string
        return str(value)
    async def _get_data_type_display_name(self, node: Node) -> str:
        """Get the display name of a node's data type"""
        if not self.opc_client:
            return "Unknown"

        try:
            dt_nodeid = await node.read_data_type()
            if dt_nodeid:
                dt_node = self.opc_client.get_node(dt_nodeid)
                dt_display_name = (await dt_node.read_display_name()).Text
                return dt_display_name or self._format_node_id(dt_nodeid)
        except Exception:
            pass
        return "Unknown"

    async def _ensure_connection(self):
        """Ensure we have a persistent connection to the OPC server"""
        if not self.opc_client:
            if not self.server_url:
                raise ValueError("OPC client not initialized and no server URL set")
            self.opc_client = Client(self.server_url)

        await self.ensure_connected_no_backoff()
        logger.info(f"Established persistent connection to {self.server_url}")

    async def init(self, server_url: str):
        """Initialize the OPC client with server URL"""
        self.server_url = server_url
        self.opc_client = Client(server_url)
        self._connected = False
        await self._ensure_connection()
        return self

    async def get_connection_status(self) -> StatusResponse:
        """Get current connection status and server info"""
        if not self.opc_client:
            return StatusResponse(connected=False, error="OPC client not initialized")

        try:
            await self._ensure_connection()
            server_info = self.opc_client.get_server_node()

            return StatusResponse(
                connected=self._connected,
                server_url=self.server_url,
                server_info=ServerInfo(
                    node_id=self._format_node_id(server_info.nodeid),
                    display_name=(await server_info.read_display_name()).Text or "",
                ),
            )
        except Exception as e:
            return StatusResponse(
                connected=False,
                server_url=self.server_url,
                error=str(e),
            )

    async def browse_root(self) -> list[NodeInfo]:
        """Browse the root nodes of the OPC server"""
        if not self.opc_client:
            raise ValueError("OPC client not initialized")

        await self._ensure_connection()
        root_node = self.opc_client.get_root_node()
        children = await root_node.get_children()

        nodes = []
        for child in children:
            display_name = (await child.read_display_name()).Text or ""
            node_id = self._format_node_id(child.nodeid)
            node_class = await child.read_node_class()
            nodes.append(NodeInfo(
                node_id=node_id,
                display_name=display_name,
                node_class=get_node_class_name(node_class.value),
            ))
        return nodes

    async def browse_node(self, node_id: str) -> list[NodeInfo]:
        """Browse children of a specific node"""
        if not self.opc_client:
            raise ValueError("OPC client not initialized")

        await self._ensure_connection()
        node = self.opc_client.get_node(node_id)
        children = await node.get_children()

        nodes = []
        for child in children:
            display_name = (await child.read_display_name()).Text or ""
            child_node_id = self._format_node_id(child.nodeid)
            node_class = await child.read_node_class()
            nodes.append(NodeInfo(
                node_id=child_node_id,
                display_name=display_name,
                node_class=get_node_class_name(node_class.value),
            ))
        return nodes

    async def get_node_details(self, node_id: str) -> NodeDetails:
        """Get detailed information about a specific node"""
        if not self.opc_client:
            raise ValueError("OPC client not initialized")

        await self._ensure_connection()
        node = self.opc_client.get_node(node_id)

        display_name = (await node.read_display_name()).Text or ""
        node_class = get_node_class_name((await node.read_node_class()).value)

        # Get optional attributes
        data_type = None
        try:
            data_type = await self._get_data_type_display_name(node)
        except Exception:
            pass

        value = None
        try:
            if node_class == "Variable":
                value = await node.read_value()
        except Exception:
            pass

        description = None
        try:
            desc = await node.read_description()
            description = desc.Text if desc.Text else None
        except Exception:
            pass

        return NodeDetails(
            node_id=node_id,
            display_name=display_name,
            node_class=node_class,
            data_type=data_type,
            value=value,
            description=description,
        )

    async def read_node_value(self, node_id: str):
        """Read current value of a variable node"""
        if not self.opc_client:
            raise ValueError("OPC client not initialized")

        await self._ensure_connection()
        node = self.opc_client.get_node(node_id)

        # Check if node is a variable
        node_class = await node.read_node_class()
        if node_class != NodeClass.Variable:
            raise ValueError(f"Node {node_id} is not a variable (node class: {node_class.name})")

        value = await node.read_value()
        return {
            "node_id": node_id,
            "value": value,
            "timestamp": None,  # Could add server timestamp if needed
        }

    async def write_node_value(self, node_id: str, value: str):
        """Write value to a variable node"""
        if not self.opc_client:
            raise ValueError("OPC client not initialized")

        await self._ensure_connection()
        node = self.opc_client.get_node(node_id)

        # Check if node is a variable
        node_class = await node.read_node_class()
        if node_class != NodeClass.Variable:
            raise ValueError(f"Node {node_id} is not a variable (node class: {node_class.name})")

        # Read the current value to check if it's an array
        current_value = await node.read_value()
        is_array = isinstance(current_value, list)

        # Read the data type and cast the value appropriately
        dt_display_name = await self._get_data_type_display_name(node)

        # Cast value based on data type
        if is_array:
            cast_value = await self._cast_array_value(value, dt_display_name)
        else:
            cast_value = await self._cast_single_value(value, dt_display_name)

        await node.write_value(cast_value)

        return {
            "node_id": node_id,
            "value": cast_value,
            "data_type": dt_display_name,
            "status": "written",
            "message": f"Successfully wrote value '{cast_value}' to node {node_id}",
        }
