from collections import deque

from lib_utils.messages.base_event_bus import BaseEventBus, Callback

from corerl.messages.events import Event, EventTopic, EventType
from corerl.messages.factory import EventBusConfig


class EventBus(BaseEventBus[Event, EventTopic, EventType]):
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

    # Overriding to have the default EventTopic
    def emit_event(self, event: Event | EventType, topic: EventTopic = EventTopic.debug_app):
        return super().emit_event(event, topic)


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
