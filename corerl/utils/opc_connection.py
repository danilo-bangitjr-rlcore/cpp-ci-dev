import asyncio
import datetime
import logging

import numpy as np
from asyncua import Client, Node, ua

_logger = logging.getLogger(__name__)

PREFIX = "opctest"


class OpcConnection:
    """Wrapper class around the asyncua library to make life easier for the rlai group"""

    def __init__(self, cfg) -> None:
        self.ip_address = cfg.ip_address
        self.port = cfg.port
        self.conn_attempts = 0
        self.values_read = 0
        self.conn_stats = cfg.conn_stats
        self.vendor = cfg.vendor if hasattr(cfg, "vendor") else "kepware"

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

    async def get_nodes(self, addresses: list[str]) -> list[Node]:
        """Returns an array of OPC nodes corresponding to the string addresses"""
        while True:
            try:
                if self.vendor == "kepware":
                    ns = 2
                else:
                    ns = 1
                nodes = [
                    self.client.get_node(f"ns={ns};s={address}") for address in addresses
                ]
                return nodes
            except (ConnectionError, ua.UaError, TimeoutError):
                _logger.warning("READ: Reconnecting in 2 seconds")
                self.update_stats()
                await asyncio.sleep(2)
                await self.reconnect()

    async def read_values(self, nodes: list[Node]) -> list[any]:
        """Reads the values of an array of OPC nodes and returns those values"""
        while True:
            try:
                return await self.client.read_values(nodes)
            except (ConnectionError, ua.UaError, TimeoutError):
                _logger.warning("READ: Reconnecting in 2 seconds")
                self.update_stats()
                await asyncio.sleep(2)
                await self.reconnect()

    async def read_variant_types(self, nodes: list[Node]) -> list[ua.VariantType]:
        """Determines the VariantType of the Values of the list of Nodes"""
        while True:
            try:
                node_ids = [node.nodeid for node in nodes]
                values = await self.client.uaclient.read_attributes(
                    node_ids, ua.AttributeIds.Value
                )
                return [value.Value.VariantType for value in values]
            except (ConnectionError, ua.UaError, TimeoutError):
                _logger.warning("READ: Reconnecting in 2 seconds")
                self.update_stats()
                await asyncio.sleep(2)
                await self.reconnect()

    async def read_attributes(
        self, nodes: list[Node], attr: ua.AttributeIds
    ) -> list[any]:
        """Returns the specified Attribute of the list of Nodes"""
        while True:
            try:
                node_ids = [node.nodeid for node in nodes]
                return await self.client.uaclient.read_attributes(node_ids, attr)
            except (ConnectionError, ua.UaError, TimeoutError):
                _logger.warning("READ: Reconnecting in 2 seconds")
                self.update_stats()
                await asyncio.sleep(2)

    async def write_values(
        self,
        nodes: list[Node],
        variants: list[ua.VariantType],
        values: list[any] | np.ndarray,
    ) -> None:
        """Writes the Values with corresponding VariantTypes to the list of Nodes"""
        while True:
            try:
                # creating explicit DataValues like this, with just a variant and value
                # avoids issues of time stamp syncing, should investigate this later
                data_values = []
                for i in range(0, len(values)):
                    data_values.append(ua.DataValue(ua.Variant(values[i], variants[i])))

                await self.client.write_values(nodes, data_values)
                return
            except (ConnectionError, ua.UaError, TimeoutError) as err:
                _logger.warning(f"{err}")
                _logger.warning("WRITE: Reconnecting in 2 seconds")
                self.update_stats()
                await asyncio.sleep(2)
                await self.reconnect()

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
