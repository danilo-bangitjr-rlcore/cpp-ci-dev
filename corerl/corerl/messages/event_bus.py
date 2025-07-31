from collections import deque

from lib_utils.messages.base_event_bus import BaseEventBus, Callback

from corerl.messages.events import RLEvent, RLEventTopic, RLEventType
from corerl.messages.factory import EventBusConfig


class EventBus(BaseEventBus[RLEvent, RLEventTopic, RLEventType]):
    """EventBus enables asynchronous communication through a ZMQ pub-sub messaging pattern.
    Spins up the scheduler thread, the consumer thread, and the FIFO subscriber queue.
    """
    def __init__(self, cfg_event_bus: EventBusConfig):
        self.cfg_event_bus = cfg_event_bus
        super().__init__(
            event_class=RLEvent,
            topic=RLEventTopic.corerl,
            consumer_name="corerl_event_bus_consumer",
            subscriber_addrs=[
                cfg_event_bus.app_connection,
                cfg_event_bus.cli_connection,
            ],
            publisher_addr=cfg_event_bus.app_connection,
        )

    # Overriding to have the default EventTopic
    def emit_event(self, event: RLEvent | RLEventType, topic: RLEventTopic = RLEventTopic.debug_app):
        return super().emit_event(event, topic)


class DummyEventBus:
    def __init__(self, queue_size: int = 10):
        self._queue = deque[RLEvent](maxlen=queue_size)

    def listen_forever(self):
        while True:
            yield

    def emit_event(self, event: RLEvent | RLEventType, topic: RLEventTopic = RLEventTopic.debug_app):
        if isinstance(event, RLEventType):
            event = RLEvent(type=event)

        self._queue.append(event)

    def get_last_events(self, n: int | None = None):
        if n is None:
            return list(self._queue)

        return [self._queue[-i] for i in range(1, n + 1)]

    def start(self): ...
    def attach_callback(self, event_type: RLEventType, cb: Callback): ...
    def attach_callbacks(self, cbs: dict[RLEventType, Callback]): ...
    def cleanup(self): ...
