import uuid
import asyncio
import logging
from collections import defaultdict
from pydantic import ValidationError
from websockets import ConnectionClosed
from websockets.asyncio.server import serve, ServerConnection, Server
from corerl.messages.events import Event, EventType, SubscribeEvent


logger = logging.getLogger(__name__)


class WebsocketServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self._server: Server | None = None
        self._serve_future: asyncio.Future[None] | None = None

        self._id_to_socket: dict[uuid.UUID, ServerConnection] = {}
        self._event_to_socket_id: dict[EventType, set[uuid.UUID]] = defaultdict(set)
        self._socket_id_to_events: dict[uuid.UUID, set[EventType]] = defaultdict(set)

    # ----------------
    # -- Server API --
    # ----------------
    async def start(self):
        assert self._server is None

        self._server = await serve(
            self._handler,
            self.host,
            self.port,
        )

        self._serve_future = asyncio.ensure_future(
            self._server.serve_forever(),
        )

    async def serve_forever(self):
        assert self._serve_future is not None
        await self._serve_future


    async def close(self):
        assert self._server is not None
        assert self._serve_future is not None

        self._serve_future.cancel()
        self._server.close()


    def connected_clients(self) -> set[uuid.UUID]:
        return set(self._id_to_socket.keys())


    # -------------------------
    # -- Server control loop --
    # -------------------------
    async def _handler(self, websocket: ServerConnection):
        self._id_to_socket[websocket.id] = websocket

        while True:
            try:
                message = await websocket.recv(decode=False)
            except ConnectionClosed:
                await self._handle_disconnect(websocket)
                break

            event = maybe_parse_event(message)
            if event is None:
                continue

            elif event.type == EventType.subscribe:
                self._handle_subscribe(message, websocket)

            else:
                await self._handle_notify(message, event.type)


    # --------------------------
    # -- Connection Lifecycle --
    # --------------------------
    def _handle_subscribe(self, msg: str | bytes, websocket: ServerConnection):
        event = SubscribeEvent.model_validate_json(msg)
        self._subscribe_to_event(event.subscribe_to, websocket.id)

        logger.debug(f'Adding subscriber to event type: <{event.subscribe_to}>')


    async def _handle_notify(self, msg: str | bytes, type: EventType):
        subscribers = self._event_to_socket_id[type]
        await asyncio.gather(*[
            self._id_to_socket[sub].send(msg) for sub in subscribers
        ])

        logger.debug(f'Received event type: <{type}>. Notifying {len(subscribers)} subscribers.')


    async def _handle_disconnect(self, websocket: ServerConnection):
        events = self._socket_id_to_events[websocket.id]
        for e in events:
            self._event_to_socket_id[e].discard(websocket.id)

        del self._id_to_socket[websocket.id]
        logger.info(f'Client disconnected: <{websocket.id}>. Unsubscribing from events: <{events}>.')


    # -------------------------
    # -- Server internal API --
    # -------------------------
    def _subscribe_to_event(self, type: EventType, socket_id: uuid.UUID):
        # create two way binding to make querying by websocket cheap
        #   e.g. for unsubscribe
        # and to make querying by event type cheap
        #   e.g. for notifications
        self._event_to_socket_id[type].add(socket_id)
        self._socket_id_to_events[socket_id].add(type)



# ---------------------
# -- Utility methods --
# ---------------------
def maybe_parse_event(msg: str | bytes) -> Event | None:
    try:
        return Event.model_validate_json(msg)
    except ValidationError:
        logger.exception('Failed to parse websocket message')
        return None
