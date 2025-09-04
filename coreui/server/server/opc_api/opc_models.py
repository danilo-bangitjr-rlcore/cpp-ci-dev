from typing import Any

from pydantic import BaseModel


class NodeInfo(BaseModel):
    node_id: str
    display_name: str
    node_class: str


class ServerInfo(BaseModel):
    node_id: str
    display_name: str


class StatusResponse(BaseModel):
    connected: bool
    server_url: str | None = None
    message: str | None = None
    error: str | None = None
    server_info: ServerInfo | None = None


class NodeDetails(BaseModel):
    node_id: str
    display_name: str
    node_class: str
    data_type: str | None = None
    value: Any | None = None
    description: str | None = None
