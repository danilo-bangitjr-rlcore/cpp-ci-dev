from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from lib_utils.messages.base_event_bus import BaseEventBus

from corerl.messages.events import Event, EventTopic, EventType
from corerl.messages.factory import EventBusConfig

Callback = Callable[[Event], Any]


class EventBus(BaseEventBus[Event]):
    """EventBus enables asynchronous communication through a ZMQ pub-sub messaging pattern.
    Spins up the scheduler thread, the consumer thread, and the FIFO subscriber queue.
    """
    def __init__(self, cfg_event_bus: EventBusConfig):
        self.cfg_event_bus = cfg_event_bus
        super().__init__(
            event_class=Event,
            topic=EventTopic.corerl,
            consumer_name="corerl_event_bus_consumer",
            subscriber_sockets=[
                cfg_event_bus.app_connection,
                cfg_event_bus.cli_connection,
            ],
            publisher_socket=cfg_event_bus.app_connection,
        )
        self._callbacks: dict[EventType, list[Callback]] = defaultdict(list)

    def emit_event(self, event: Event | EventType, topic: EventTopic = EventTopic.debug_app):
        if isinstance(event, EventType):
            event = Event(type=event)

        message_data = event.model_dump_json()
        self.publisher_socket.send_string(f"{topic} {message_data}")


    def listen_forever(self):
        while True:
            event: Event | None = self.recv_event()
            if event is None:
                continue

            for cb in self._callbacks[event.type]:
                cb(event)

            yield event


    def attach_callback(self, event_type: EventType, cb: Callback):
        self._callbacks[event_type].append(cb)


    def attach_callbacks(self, cbs: dict[EventType, Callback]):
        for event_type, cb in cbs.items():
            self.attach_callback(event_type, cb)

class DummyEventBus:
    def __init__(self, queue_size: int = 10):
        self._queue = deque[Event](maxlen=queue_size)

    def listen_forever(self):
        while True:
            yield

    def emit_event(self, event: Event | EventType, topic: EventTopic = EventTopic.debug_app):
        if isinstance(event, EventType):
            event = Event(type=event)

        self._queue.append(event)

    def get_last_events(self, n: int | None = None):
        if n is None:
            return list(self._queue)

        return [self._queue[-i] for i in range(1, n + 1)]

    def start(self): ...
    def attach_callback(self, event_type: EventType, cb: Callback): ...
    def attach_callbacks(self, cbs: dict[EventType, Callback]): ...
    def cleanup(self): ...
