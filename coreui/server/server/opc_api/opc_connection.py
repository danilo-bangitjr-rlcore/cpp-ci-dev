import logging
from typing import Any

from asyncua import Client
from lib_utils.opc.opc_communication import OPC_Connection

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

    async def get_connection_status(self) -> dict[str, Any]:
        """Get current connection status and server info"""
        if not self.opc_client:
            return {
                "connected": False,
                "server_url": None,
                "error": "OPC client not initialized",
            }

        try:
            # Ensure we're connected before checking connection status
            await self.ensure_connected_no_backoff()

            # Now safely check connection and get server info
            await self.opc_client.check_connection()
            server_info = self.opc_client.get_server_node()

            return {
                "connected": self._connected,
                "server_url": self.server_url,
                "server_info": {
                    "node_id": str(server_info.nodeid),
                    "display_name": (await server_info.read_display_name()).Text,
                },
            }
        except Exception as e:
            return {
                "connected": False,
                "server_url": self.server_url,
                "error": str(e),
            }
        finally:
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e!s}")
