from fastapi import APIRouter, HTTPException

from server.opc_api.opc_connection import OPC_Connection_UI
from server.opc_api.opc_models import NodeDetails, NodeInfo, StatusResponse

opc_connection = OPC_Connection_UI()
opc_router = APIRouter()


# Core Connection Management

@opc_router.post("/connect")
async def connect_to_server(url: str):
    """Establish persistent connection to OPC server"""
    try:
        await opc_connection.init(url)
        return {
            "status": "connected",
            "server_url": url,
            "message": "Successfully established persistent connection to OPC server",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect to OPC server: {e!s}") from e


@opc_router.post("/disconnect")
async def disconnect_from_server():
    """Disconnect from OPC server and cleanup connection"""
    try:
        await opc_connection.cleanup()
        opc_connection.server_url = ""
        opc_connection._connected = False
        return {
            "status": "disconnected",
            "message": "Successfully disconnected from OPC server",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to disconnect: {e!s}") from e


@opc_router.get("/status")
async def get_connection_status() -> StatusResponse:
    """Get current connection status"""
    if not opc_connection.server_url:
        return StatusResponse(connected=False, message="No OPC server URL configured")

    try:
        return await opc_connection.get_connection_status()
    except Exception as e:
        return StatusResponse(
            connected=False,
            server_url=opc_connection.server_url,
            error=str(e),
        )


# Node Browsing & Navigation

@opc_router.get("/browse")
async def browse_root() -> list[NodeInfo]:
    """Get root nodes of the OPC server"""
    if not opc_connection.server_url:
        raise HTTPException(status_code=400, detail="No OPC server URL configured")

    try:
        return await opc_connection.browse_root()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to browse root nodes: {e!s}") from e


@opc_router.get("/browse/{node_id}")
async def browse_node(node_id: str) -> list[NodeInfo]:
    """Browse children of a specific node"""
    if not opc_connection.server_url:
        raise HTTPException(status_code=400, detail="No OPC server URL configured")

    try:
        return await opc_connection.browse_node(node_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to browse node {node_id}: {e!s}") from e


@opc_router.get("/node/{node_id}")
async def get_node_details(node_id: str) -> NodeDetails:
    """Get detailed information about a specific node"""
    if not opc_connection.server_url:
        raise HTTPException(status_code=400, detail="No OPC server URL configured")

    try:
        return await opc_connection.get_node_details(node_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get details for node {node_id}: {e!s}") from e


# Data Access

@opc_router.get("/read/{node_id}")
async def read_node_value(node_id: str):
    """Read current value of a variable node"""


@opc_router.post("/write/{node_id}")
async def write_node_value(node_id: str, value: str):
    """Write value to a variable node"""


# Enhanced Features (placeholders)

@opc_router.get("/search")
async def search_nodes(query: str, node_class: str | None = None):
    """Search for nodes by name or other criteria"""


@opc_router.get("/attributes/{node_id}")
async def get_node_attributes(node_id: str):
    """Get all attributes of a node (data type, access level, etc.)"""
