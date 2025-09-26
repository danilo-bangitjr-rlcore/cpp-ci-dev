import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType
from typing import Any
import asyncio

import backoff
from asyncua import Client, Node, ua
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 30

def log_backoff(details: Any):
    wait = details["wait"]
    tries = details["tries"]
    func = details["target"].__name__
    logger.error(f"Backing off {wait:.1f} seconds after {tries} tries calling {func}")


class NodeData(BaseModel):
    node: Node
    var_type: ua.VariantType
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OPC_Connection(ABC):
    def __init__(self):
        self.opc_client: Client | None = None
        self.registered_nodes: dict[str, NodeData] = {}
        self._context_active = False
        self.connection_id = ""
        self._connected = False

    # -------------------- #
    # --- Init methods --- #
    # -------------------- #

    @abstractmethod
    async def init(self, *args: Any, **kwargs: Any) -> Any:
        pass

    # -------------------------- #
    # --- Manage Connections --- #
    # -------------------------- #

    @backoff.on_exception(backoff.expo, Exception, max_value=MAX_BACKOFF_SECONDS, on_backoff=log_backoff)
    async def ensure_connected(self):
        """Ensure connected with backoff; log intention."""
        logger.debug(f"About to ensure connection with backoff for {self.connection_id}")
        await self.ensure_connected_no_backoff()

    async def ensure_connected_no_backoff(self):
        assert self.opc_client is not None, 'OPC client is not initialized'

        if self._connected is False:
            logger.debug(f"Attempting direct connect to OPC client for {self.connection_id}")
            await self.opc_client.connect()
            self._connected = True

        try:
            logger.debug(f"Checking OPC client connection health for {self.connection_id}")
            await self.opc_client.check_connection()

        except Exception as e:
            logger.warning(f"Retrying connection to {self.connection_id}.\n{e!s}")
            self._connected = False
            await self.opc_client.disconnect()
            await asyncio.sleep(0.1)
            await self.opc_client.connect()
            await self.opc_client.check_connection()
            self._connected = True

        return self.opc_client

    async def start(self):
        logger.debug(f"Starting OPC connection wrapper for {self.connection_id}")
        await self.ensure_connected()
        return self

    async def cleanup(self):
        logger.debug(f"About to cleanup OPC connection {self.connection_id}")
        if self.opc_client is None:
            logger.debug(f"No OPC client to cleanup for {self.connection_id}")
            return self

        await self.opc_client.disconnect()
        self._connected = False
        logger.debug(f"Cleanup complete for {self.connection_id}")
        return self

    async def __aenter__(self):
        logger.debug(f"Entering OPC context for {self.connection_id}")
        self._context_active = True
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ):
        _ = exc_type, exc, tb
        logger.debug(f"Exiting OPC context for {self.connection_id} (exc={exc})")
        self._context_active = False
        await self.cleanup()

    @staticmethod
    def requires_context(func: Callable[..., Any]):
        """Decorator that ensures method is called within active context"""
        async def wrapper(self: OPC_Connection, *args: Any, **kwargs: Any):
            logger.debug(f"Checking OPC context active for {self.connection_id} before calling {func.__name__}")
            if not self._context_active:
                raise RuntimeError(f"Function {func.__name__} must be called within the OPC context manager")
            return await func(self, *args, **kwargs)

        return wrapper
