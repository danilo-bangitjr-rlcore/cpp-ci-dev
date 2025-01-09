from __future__ import annotations
import asyncio
from collections.abc import Callable, Coroutine
import datetime
import logging
from typing import Any, Concatenate
from uuid import UUID

import numpy as np
from asyncua import Client, Node, ua
from asyncua.sync import Client as SyncClient, SyncNode
from corerl.configs.config import MISSING, config

_logger = logging.getLogger(__name__)

PREFIX = "opctest"


def make_opc_node_id(node_id: str | int | bytes | UUID, namespace: str | int = 0):
    if isinstance(node_id, int):
        return f"ns={namespace};i={node_id}"
    elif isinstance(node_id, UUID):
        return f"ns={namespace};g={node_id}"
    elif isinstance(node_id, bytes):
        return f"ns={namespace};b={node_id}"
    else:
        return f"ns={namespace};s={node_id}"


def sync_browse_opc_nodes(client: SyncClient, node: SyncNode):
    """Recursively browse OPC nodes"""
    structure = {}
    children = node.get_children()

    for child in children:
        child_node_class = child.read_node_class()
        child_browse_name = child.read_browse_name()
        child_display_name = child_browse_name.Name

        # If object, recurse into it
        if child_node_class == ua.NodeClass.Object:
            structure[child_display_name] = sync_browse_opc_nodes(client, child)

        # Elif variable, write value and data type
        elif child_node_class == ua.NodeClass.Variable:
            value = child.read_value()
            data_type_node = child.read_data_type()
            data_type_name = client.get_node(data_type_node).read_display_name()

            structure[child_display_name] = {
                "val": value,
                "DataType": data_type_name.Text,
                "NameSpace": child.nodeid.NamespaceIndex,
                "Identifier": child.nodeid.Identifier,
                "nodeid": make_opc_node_id(child.nodeid.Identifier, child.nodeid.NamespaceIndex)
            }

    return structure


def linear_backoff(direction: str, attempts: int = 50):
    def _inner[T, **P](f: Callable[Concatenate[OpcConnection, P], Coroutine[None, None, T]]):
        async def wrapper(self: OpcConnection, *args: P.args, **kwargs: P.kwargs) -> T:

            for _ in range(attempts):
                try:
                    return await f(self, *args, **kwargs)

                except (ConnectionError, ua.UaError, TimeoutError) as err:
                    _logger.warning(f"{err}")
                    _logger.warning(f"{direction.upper()}: Reconnecting in {self._conn_timeout} seconds")
                    self.update_stats()
                    await self.reconnect()
                    await asyncio.sleep(self._conn_timeout)

            raise Exception(f'Failed to reconnect after {attempts} attempts')

        return wrapper
    return _inner


@config()
class OpcConfig:
    ip_address: str = MISSING
    port: int = MISSING
    vendor: str = 'kepware'

    timeout: float = 2.
    conn_stats: bool = True


class OpcConnection:
    """Wrapper class around the asyncua library to make life easier for the rlai group"""

    def __init__(self, cfg: OpcConfig):
        self.ip_address = cfg.ip_address
        self.port = cfg.port
        self.conn_attempts = 0
        self.values_read = 0
        self.conn_stats = cfg.conn_stats
        self._conn_timeout = cfg.timeout
        self.vendor = cfg.vendor

    async def connect(self) -> None:
        """Connects to the OPC server and opens any log files"""
        file_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stats_fname = f"{PREFIX}_{file_datetime}.stats"
        if self.conn_stats:
            try:
                self.stats = open(stats_fname, mode="w")
            except OSError as err:
                _logger.error(f"Could not open stats file: {stats_fname} for writing")
                _logger.error("Shutting down")
                raise err

        if self.vendor == "kepware":
            opc_url = f"opc.tcp://{self.ip_address}:{self.port}/PTC/ThingWorxKepwareServer/"
        elif self.vendor == "ignition":
            opc_url = f"opc.tcp://{self.ip_address}:{self.port}"
        else:
            raise NotImplementedError

        self.client = Client(
            url=opc_url,
            timeout=20,
            watchdog_intervall=30,
        )
        await self.client.connect()

    @linear_backoff(direction='read')
    async def get_nodes(self, addresses: list[str]) -> list[Node]:
        """Returns an array of OPC nodes corresponding to the string addresses"""
        if self.vendor == "kepware":
            ns = 2
        else:
            ns = 1
        nodes = [
            self.client.get_node(f"ns={ns};s={address}") for address in addresses
        ]
        return nodes

    @linear_backoff(direction='read')
    async def read_values(self, nodes: list[Node]) -> list[Any]:
        """Reads the values of an array of OPC nodes and returns those values"""
        return await self.client.read_values(nodes)

    @linear_backoff(direction='read')
    async def read_variant_types(self, nodes: list[Node]) -> list[ua.VariantType]:
        """Determines the VariantType of the Values of the list of Nodes"""
        node_ids = [node.nodeid for node in nodes]
        values = await self.client.uaclient.read_attributes(
            node_ids, ua.AttributeIds.Value
        )
        return [value.Value.VariantType for value in values if value.Value is not None]

    @linear_backoff(direction='read')
    async def read_attributes(
        self, nodes: list[Node], attr: ua.AttributeIds
    ) -> list[Any]:
        """Returns the specified Attribute of the list of Nodes"""
        node_ids = [node.nodeid for node in nodes]
        return await self.client.uaclient.read_attributes(node_ids, attr)

    @linear_backoff(direction='write')
    async def write_values(
        self,
        nodes: list[Node],
        variants: list[ua.VariantType],
        values: list[Any] | np.ndarray,
    ) -> None:
        """Writes the Values with corresponding VariantTypes to the list of Nodes"""
        # creating explicit DataValues like this, with just a variant and value
        # avoids issues of time stamp syncing, should investigate this later
        data_values = []
        for i in range(0, len(values)):
            data_values.append(ua.DataValue(ua.Variant(values[i], variants[i])))

        await self.client.write_values(nodes, data_values)

    async def disconnect(self) -> None:
        """Disconnects from the OPC server and closes any log files"""
        if self.conn_stats:
            try:
                self.stats.flush()
                self.stats.close()
            except OSError:
                _logger.error("Could not close stats file")

        await self.client.disconnect()

    async def reconnect(self) -> None:
        # await self.disconnect()
        await self.connect()

    def update_stats(self) -> None:
        if self.conn_stats:
            self.stats.seek(0)
            self.stats.write(
                f"con_attempts={self.conn_attempts}\nreads={self.values_read}"
            )
