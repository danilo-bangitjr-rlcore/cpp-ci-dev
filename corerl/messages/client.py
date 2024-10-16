import asyncio
from collections import defaultdict
from collections.abc import Callable
import logging
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Protocol
from websockets import ConnectionClosed
from websockets.asyncio.client import connect, ClientConnection

from corerl.messages.events import Event, EventType, SubscribeEvent, maybe_parse_event
from corerl.utils.asyncio import MaybeAwaitable, maybe_await


logger = logging.getLogger(__name__)

Callback = Callable[[Event], MaybeAwaitable[None]]

# ----------------
# -- Websockets --
# ----------------
class WebsocketClient:
    def __init__(self, host: str, port: int | str):
        self._host = host
        self._port = port

        self._closed = False
        self._websocket: ClientConnection | None = None
        self._first_connect_future: asyncio.Future | None = None
        self._reconnect_future: asyncio.Future | None = None
        self._loop: AbstractEventLoop | None = None
        self._pool: ThreadPoolExecutor | None = None

        self._callbacks: dict[EventType, list[Callback]] = defaultdict(list)


    # ---------------
    # -- Async API --
    # ---------------
    async def start(self, max_attempts: int = -1):
        self._loop = asyncio.get_event_loop()
        self._first_connect_future = asyncio.Future(loop=self._loop)
        self._reconnect_future = asyncio.ensure_future(
            self._continuously_reconnect(max_attempts),
            loop=self._loop,
        )


    async def close(self):
        if self._closed:
            return

        self._closed = True
        maybe_cancel(self._reconnect_future)
        await maybe_close(self._websocket)


    async def subscribe(
        self,
        to: EventType,
        cb: Callable[[Event], MaybeAwaitable[None]],
    ):
        sub_event = SubscribeEvent(subscribe_to=to)
        await self.emit_event(sub_event)

        self._callbacks[to].append(cb)

    async def send_message(self, msg: str):
        try:
            socket = await asyncio.wait_for(
                asyncio.shield(self.ensure_connected()),
                timeout=1,
            )
            await socket.send(msg)
        except Exception:
            logging.error(f'Failed to send message <{msg}>')


    async def emit_event(self, event: Event | EventType):
        # allow a common pattern of emiting a zero arg event:
        #   client.emit_event(EventType.heartbeat)
        if isinstance(event, EventType):
            event = Event(type=event)

        await self.send_message(event.model_dump_json())


    async def ensure_connected(self):
        assert self._first_connect_future is not None
        await self._first_connect_future

        assert self._websocket is not None
        return self._websocket


    async def listen_forever(self):
        assert self._reconnect_future is not None
        await self._reconnect_future

    # --------------
    # -- Sync API --
    # --------------
    def start_sync(self):
        self._sync = True

        def _run_forever():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.start())

            assert self._reconnect_future is not None
            loop.run_until_complete(self._reconnect_future)

        self._pool = ThreadPoolExecutor(1)
        self._pool.submit(_run_forever)


    def close_sync(self):
        if self._closed:
            return

        self._closed = True
        maybe_cancel(self._reconnect_future)

        assert self._pool is not None
        self._pool.shutdown(wait=False, cancel_futures=True)


    def emit_event_sync(self, event: Event | EventType):
        loop = self._loop
        if loop is None:
            loop = self.restart_sync()

        fut = self.emit_event(event)
        fut = asyncio.run_coroutine_threadsafe(fut, loop)


    def restart_sync(self):
        self.close_sync()
        self.start_sync()

        assert self._loop is not None
        return self._loop


    # ------------------------
    # -- Internal Heartbeat --
    # ------------------------
    async def _continuously_reconnect(self, max_attempts: int = -1):
        assert self._first_connect_future is not None

        connector = connect(
            f'ws://{self._host}:{self._port}',
            open_timeout=1,
        )

        attempts = 0
        async for socket in connector:
            self._websocket = socket
            attempts += 1

            if not self._first_connect_future.done():
                self._first_connect_future.set_result(None)

            try:
                await self._listen_forever(socket)
            except ConnectionClosed:
                logging.exception('Connection closed unexpectedly')

            if max_attempts != -1 and attempts >= max_attempts:
                break


    async def _listen_forever(self, socket: ClientConnection):
        while True:
            msg = await socket.recv(decode=False)
            logging.debug(f'Received msg: <{msg}>')
            await self._handle_msg(msg)


    async def _handle_msg(self, msg: str | bytes):
        event = maybe_parse_event(msg)
        if event is None:
            return

        await asyncio.gather(*[
            maybe_await(cb(event))
            for cb in self._callbacks[event.type]
        ])

# ------------------
# -- Dummy Client --
# ------------------
class DummyWebsocketClient(WebsocketClient):
    def __init__(self, host: str, port: int | str):
        super().__init__(host, port)

    async def start(self, max_attempts: int = -1):
        ...

    def start_sync(self):
        ...

    async def close(self):
        ...

    async def send_message(self, msg: str):
        ...

    async def emit_event(self, event: Event | EventType):
        ...

    def emit_event_sync(self, event: Event | EventType):
        ...


# -----------------
# -- Message Bus --
# -----------------
class MessageBusClientConfig(Protocol):
    enable: bool
    host: str
    port: int | str


def make_msg_bus_client(cfg: MessageBusClientConfig) -> WebsocketClient:
    if cfg.enable:
        return WebsocketClient(cfg.host, cfg.port)

    return DummyWebsocketClient('', '')


# --------------------
# -- Internal utils --
# --------------------
def maybe_cancel(e: asyncio.Future | Future | None):
    if e is None:
        return

    e.cancel()

def maybe_shutdown(e: ThreadPoolExecutor | None):
    if e is None:
        return

    e.shutdown(cancel_futures=True)

async def maybe_close(w: ClientConnection | None):
    if w is None:
        return

    await w.close()
