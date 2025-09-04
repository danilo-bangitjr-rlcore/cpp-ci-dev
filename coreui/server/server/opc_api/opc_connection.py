import logging

from asyncua import Client
from lib_utils.opc.opc_communication import OPC_Connection

from server.opc_api.opc_models import NodeDetails, NodeInfo, ServerInfo, StatusResponse

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 30


class OPC_Connection_UI(OPC_Connection):
    def __init__(self):
        super().__init__()
        self.server_url: str = ""
        self._connected = False

    async def init(self, server_url: str):
        """Initialize the OPC client with server URL"""
        self.server_url = server_url
        self.opc_client = Client(server_url)
        self._connected = False

        # Test connection
        await self.ensure_connected_no_backoff()
        await self.cleanup()
        return self

    async def get_connection_status(self) -> StatusResponse:
        """Get current connection status and server info"""
        if not self.opc_client:
            return StatusResponse(
                connected=False,
                error="OPC client not initialized",
            )

        try:
            # Ensure we're connected before checking connection status
            await self.ensure_connected_no_backoff()

            # Now safely check connection and get server info
            await self.opc_client.check_connection()
            server_info = self.opc_client.get_server_node()

            return StatusResponse(
                connected=self._connected,
                server_url=self.server_url,
                server_info=ServerInfo(
                    node_id=str(server_info.nodeid),
                    display_name=(await server_info.read_display_name()).Text or "",
                ),
            )
        except Exception as e:
            return StatusResponse(
                connected=False,
                server_url=self.server_url,
                error=str(e),
            )
        finally:
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e!s}")

    async def browse_root(self) -> list[NodeInfo]:
        """Browse the root nodes of the OPC server"""
        if not self.opc_client:
            raise ValueError("OPC client not initialized")

        try:
            await self.ensure_connected_no_backoff()
            root_node = self.opc_client.get_root_node()
            children = await root_node.get_children()
            nodes = []
            for child in children:
                display_name = (await child.read_display_name()).Text or ""
                node_id = str(child.nodeid)
                node_class = await child.read_node_class()
                nodes.append(NodeInfo(
                    node_id=node_id,
                    display_name=display_name,
                    node_class=str(node_class),
                ))
            return nodes
        except Exception as e:
            logger.error(f"Error browsing root nodes: {e!s}")
            raise
        finally:
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e!s}")

    async def browse_node(self, node_id: str) -> list[NodeInfo]:
        """Browse children of a specific node"""
        if not self.opc_client:
            raise ValueError("OPC client not initialized")

        try:
            await self.ensure_connected_no_backoff()
            node = self.opc_client.get_node(node_id)
            children = await node.get_children()
            nodes = []
            for child in children:
                display_name = (await child.read_display_name()).Text or ""
                child_node_id = str(child.nodeid)
                node_class = await child.read_node_class()
                nodes.append(NodeInfo(
                    node_id=child_node_id,
                    display_name=display_name,
                    node_class=str(node_class),
                ))
            return nodes
        except Exception as e:
            logger.error(f"Error browsing node {node_id}: {e!s}")
            raise
        finally:
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e!s}")

    async def get_node_details(self, node_id: str) -> NodeDetails:
        """Get detailed information about a specific node"""
        if not self.opc_client:
            raise ValueError("OPC client not initialized")

        try:
            await self.ensure_connected_no_backoff()
            node = self.opc_client.get_node(node_id)

            display_name = (await node.read_display_name()).Text or ""
            node_class = str(await node.read_node_class())

            data_type = None
            try:
                dt = await node.read_data_type()
                data_type = str(dt)
            except Exception:
                pass  # Not all nodes have data type

            value = None
            try:
                if node_class == "Variable":
                    value = await node.read_value()
            except Exception:
                pass  # Value might not be readable

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
        except Exception as e:
            logger.error(f"Error getting details for node {node_id}: {e!s}")
            raise
        finally:
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e!s}")
