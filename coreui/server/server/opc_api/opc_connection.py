import logging

from asyncua import Client
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
            dt = await node.read_data_type()
            data_type = self._format_node_id(dt)
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
