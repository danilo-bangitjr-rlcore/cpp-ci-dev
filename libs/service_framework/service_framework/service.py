import asyncio
import logging
import signal
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from enum import Enum

from lib_defs.type_defs.base_events import EventTopic, EventType
from lib_events.client.event_bus_client import EventBusClient

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service lifecycle state."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class RLTuneService(ABC):
    def __init__(
        self,
        service_name: str,
        event_topic: EventTopic,
        event_bus_host: str = "localhost",
        event_bus_port: int = 5580,
        event_bus_enabled: bool = True,
    ):
        self.service_name = service_name
        self._event_topic = event_topic
        self._state = ServiceState.CREATED
        self._error: Exception | None = None
        self._event_bus_host = event_bus_host
        self._event_bus_port = event_bus_port
        self._event_bus_enabled = event_bus_enabled
        self._event_bus_client: EventBusClient | None = None
        self._shutdown_requested: list[bool] | None = None

    # ============================================================
    # Public Lifecycle API
    # ============================================================

    async def start(self) -> None:
        self._state = ServiceState.STARTING
        logger.info(f"Starting service '{self.service_name}'")

        try:
            await self._setup_event_bus_client()
            await self._do_start()

            self._state = ServiceState.RUNNING
            self._emit_lifecycle_event(EventType.service_started)
            logger.info(f"Service '{self.service_name}' started successfully")

            await self._run_loop()

        except asyncio.CancelledError:
            logger.info(f"Service '{self.service_name}' start cancelled")
            raise

        except Exception as e:
            self._state = ServiceState.FAILED
            self._error = e
            logger.exception(f"Service '{self.service_name}' failed to start: {e}")
            self._emit_lifecycle_event(EventType.service_error)
            raise

    async def stop(self) -> None:
        if self._state in (ServiceState.STOPPED, ServiceState.STOPPING):
            logger.debug(f"Service '{self.service_name}' already stopped/stopping")
            return

        self._state = ServiceState.STOPPING
        logger.info(f"Stopping service '{self.service_name}'")

        try:
            await self._do_stop()

            self._emit_lifecycle_event(EventType.service_stopped)
            await asyncio.sleep(0.2)
            await self._teardown_event_bus_client()

            self._state = ServiceState.STOPPED
            logger.info(f"Service '{self.service_name}' stopped successfully")

        except Exception as e:
            logger.exception(f"Error stopping service '{self.service_name}': {e}")
            self._state = ServiceState.FAILED
            self._error = e
            raise

    # ============================================================
    # State Query Methods
    # ============================================================

    def is_running(self) -> bool:
        return self._state == ServiceState.RUNNING

    def is_stopped(self) -> bool:
        return self._state == ServiceState.STOPPED

    def is_failed(self) -> bool:
        return self._state == ServiceState.FAILED

    def get_state(self) -> ServiceState:
        return self._state

    def get_error(self) -> Exception | None:
        return self._error

    # ============================================================
    # Event Bus Integration
    # ============================================================

    def get_event_bus_client(self) -> EventBusClient | None:
        return self._event_bus_client

    def _emit_lifecycle_event(self, event_type: EventType) -> None:
        if not self._event_bus_enabled or self._event_bus_client is None:
            return

        self._event_bus_client.emit_event(event_type, topic=self._event_topic)
        logger.debug(f"Emitted lifecycle event: {event_type}")

    async def _setup_event_bus_client(self) -> None:
        if not self._event_bus_enabled:
            return

        logger.info(
            f"Connecting event bus client: {self._event_bus_host}:{self._event_bus_port}",
        )

        client = EventBusClient(
            host=self._event_bus_host,
            port=self._event_bus_port,
        )
        client.connect()
        self._event_bus_client = client

        logger.info("Event bus client connected")

    async def _teardown_event_bus_client(self) -> None:
        if self._event_bus_client is None:
            return

        logger.info("Closing event bus client")
        self._event_bus_client.close()
        logger.debug("Event bus client closed")

    # ============================================================
    # Abstract Methods - Implement in Subclasses
    # ============================================================

    @abstractmethod
    async def _do_start(self) -> None:
        ...

    @abstractmethod
    async def _do_stop(self) -> None:
        ...

    @abstractmethod
    async def _do_run(self) -> AsyncGenerator[None]:
        ...
        yield

    async def _run_loop(self) -> None:
        async for _ in self._do_run():
            if self._state == ServiceState.STOPPING:
                break
            if self._shutdown_requested and self._shutdown_requested[0]:
                await self.stop()
                break

    # ============================================================
    # Run Forever with Retry Logic
    # ============================================================

    def run_forever(
        self,
        max_retries: int = 5,
        retry_window_hours: int = 1,
        enable_retry: bool = True,
    ) -> None:
        shutdown_requested = [False]
        sigint_received = [False]

        self._setup_signal_handlers(shutdown_requested, sigint_received)

        if not enable_retry:
            self._run_without_retry(shutdown_requested, sigint_received)
            return

        self._run_with_retry(
            shutdown_requested,
            sigint_received,
            max_retries,
            retry_window_hours,
        )

    def _setup_signal_handlers(
        self,
        shutdown_requested: list[bool],
        sigint_received: list[bool],
    ) -> None:
        def sigterm_handler(signum: int, frame: object) -> None:
            logger.info("Received SIGTERM signal")
            shutdown_requested[0] = True

        def sigint_handler(signum: int, frame: object) -> None:
            logger.info("Received SIGINT signal")
            shutdown_requested[0] = True
            sigint_received[0] = True

        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigint_handler)

    def _run_without_retry(
        self,
        shutdown_requested: list[bool],
        sigint_received: list[bool],
    ) -> None:
        try:
            asyncio.run(self._run_once(shutdown_requested, sigint_received))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            raise
        except Exception as e:
            logger.exception(f"Service failed: {e}")
            sys.exit(1)

    def _run_with_retry(
        self,
        shutdown_requested: list[bool],
        sigint_received: list[bool],
        max_retries: int,
        retry_window_hours: int,
    ) -> None:
        retries = 0
        last_error = datetime.now(UTC)

        while not shutdown_requested[0]:
            try:
                asyncio.run(self._run_once(shutdown_requested, sigint_received))
                break

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                raise

            except Exception as e:
                logger.exception(f"Service failed: {e}")

                if shutdown_requested[0]:
                    break

                retries, last_error = self._handle_retry(
                    retries,
                    last_error,
                    max_retries,
                    retry_window_hours,
                )

    def _handle_retry(
        self,
        retries: int,
        last_error: datetime,
        max_retries: int,
        retry_window_hours: int,
    ) -> tuple[int, datetime]:
        now = datetime.now(UTC)

        if now - last_error < timedelta(hours=retry_window_hours):
            retries += 1
            if retries >= max_retries:
                logger.error(
                    f"Too many retries ({max_retries}) within "
                    f"{retry_window_hours} hour(s), giving up",
                )
                sys.exit(70)

            backoff_minutes = 2 ** (retries - 1)
            logger.warning(
                f"Retry {retries}/{max_retries}, backing off "
                f"{backoff_minutes} minute(s)",
            )
            time.sleep(60 * backoff_minutes)
            last_error = now

        else:
            retries = 1
            last_error = now
            logger.info("Error window expired, resetting retry counter")

        self._state = ServiceState.CREATED
        self._error = None

        return retries, last_error

    async def _run_once(
        self,
        shutdown_requested: list[bool],
        sigint_received: list[bool],
    ) -> None:
        self._shutdown_requested = shutdown_requested

        try:
            await self.start()

        finally:
            if sigint_received[0]:
                raise KeyboardInterrupt()
