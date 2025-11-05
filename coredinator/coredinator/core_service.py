import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import uvicorn
from lib_defs.type_defs.base_events import Event, EventTopic
from lib_events.client.event_bus_client import EventBusClient
from service_framework.service import RLTuneService

from coredinator.app import create_app
from coredinator.logging_config import setup_structured_logging

logger = logging.getLogger(__name__)


class CoredinatorService(RLTuneService):
    def __init__(
        self,
        base_path: Path,
        port: int,
        log_file: Path | None,
        log_level: str,
        console_output: bool,
        event_bus_host: str,
        event_bus_port: int,
    ):
        super().__init__(
            service_name='coredinator',
            event_topic=EventTopic.coredinator,
            event_bus_host=event_bus_host,
            event_bus_port=event_bus_port,
        )
        self.base_path = base_path
        self.port = port
        self.log_file = log_file
        self.log_level = log_level
        self.console_output = console_output
        self.event_bus_host = event_bus_host
        self.event_bus_port = event_bus_port

        self.app = None
        self.server = None

    async def _do_start(self):
        setup_structured_logging(
            log_file_path=self.log_file,
            log_level=self.log_level,
            console_output=self.console_output,
        )

        logger.info(f"Creating FastAPI application at {self.base_path}, port {self.port}")

        self.app = create_app(
            self.base_path,
            self.event_bus_host,
            self.event_bus_port,
        )

        # Start the event bus server BEFORE setting up client
        logger.info("Starting event bus manager (server)")
        self.app.state.event_bus_manager.start()

        # Now set up the event bus client to connect to localhost
        # (not the bind address which may be *)
        await self._setup_event_bus_client()

        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_config=None,
        )
        self.server = uvicorn.Server(config)

        logger.info("Coredinator web server ready")

    async def _setup_event_bus_client(self):
        # Coredinator needs to connect to its own event bus for consuming events
        # Convert bind address (*) to localhost for client connection
        host = "localhost" if self.event_bus_host == "*" else self.event_bus_host

        logger.info(
            f"Connecting event bus client: {host}:{self.event_bus_port}",
        )

        self._event_bus_client = EventBusClient(
            event_class=Event,
            host=host,
            port=self.event_bus_port,
        )
        self._event_bus_client.connect()

        logger.info("Event bus client connected")

    async def _do_run(self) -> AsyncGenerator[None]:
        assert self.server is not None

        logger.info("Starting uvicorn server")
        serve_task = asyncio.create_task(self.server.serve())
        while not serve_task.done():
            await asyncio.sleep(0.1)
            yield

        await serve_task

    async def _do_stop(self):
        # Shutdown order: client -> web server -> event bus server
        await self._teardown_event_bus_client()

        if self.server is not None:
            logger.info("Shutting down uvicorn server")
            self.server.should_exit = True
            await asyncio.sleep(0.1)

        if self.app is not None:
            logger.info("Stopping event bus manager (server)")
            self.app.state.event_bus_manager.stop()
