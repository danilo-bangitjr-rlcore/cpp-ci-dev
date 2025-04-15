"""
corerl.utils.coreio
~~~~~~~~~~~~~~~~~~~

This module implements the CoreRL communications to CoreIO (ThinClient) process,
motivated by the need to remove the python dependency on the OPC-UA library.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urljoin

import backoff
import requests

logger = logging.getLogger(__name__)


class OPCUADataType(Enum):
    Null = 0
    Boolean = 1
    SByte = 2  # signed Byte = Int8
    Byte = 3  # unsigned Byte = UInt8
    Int16 = 4
    UInt16 = 5
    Int32 = 6
    UInt32 = 7
    Int64 = 8
    UInt64 = 9
    Float = 10
    Double = 11
    String = 12
    DateTime = 13
    Guid = 14
    ByteString = 15
    XmlElement = 16
    NodeId = 17
    ExpandedNodeId = 18
    StatusCode = 19
    QualifiedName = 20
    LocalizedText = 21
    ExtensionObject = 22
    DataValue = 23
    Variant = 24
    DiagnosticInfo = 25


class OPCUANodeWriteValue:
    def __init__(self, node_id: str, value: Any, data_type: Optional[OPCUADataType] = None):
        self.node_id = node_id
        self.value = value
        self.data_type = data_type

    def to_post_payload(self) -> Dict[str, Any]:
        """Convert the node write value into the POST request expected payload."""
        payload = {"nodeId": self.node_id, "value": self.value}
        if self.data_type:
            payload["dataType"] = self.data_type.value
        return payload


class CoreIOThinClient:
    def __init__(self, coreio_origin: str):
        self.coreio_origin = coreio_origin

    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException), max_time=30)
    def write_opcua_nodes(self, connection_id: str, nodes_to_write: List[OPCUANodeWriteValue]):
        """Sends an HTTP POST request to CoreIO (Thin Client) to write OPC-UA node values.

        :param connection_id: the UUID for the CoreIO configured OPC-UA server connection
        :param nodes_to_write: list of :class:`OPCUANodeWriteValue <OPCUANodeWriteValue` instances to write.
        """
        endpoint = urljoin(self.coreio_origin, f"/api/opcua/connections/{connection_id}/write")
        payload = {"nodesToWrite": [node.to_post_payload() for node in nodes_to_write]}
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException), max_time=30)
    def read_opcua_node(self, connection_id: str, node_identifier: str):
        search_params = urlencode({"rawNodeToRead": node_identifier})
        endpoint = urljoin(self.coreio_origin, f"/api/opcua/connections/{connection_id}/read?{search_params}")
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()
