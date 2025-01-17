import zmq

from corerl.eval.writer import MetricsWriter
from corerl.messages.events import Event, EventTopic, EventType


class AppState:
    def __init__(self, metrics: MetricsWriter, event_bus: zmq.Socket | None):
        self.metrics = metrics
        self.event_bus = event_bus

    def emit_event(self, event: Event | EventType, topic: EventTopic = EventTopic.app):
        if not self.event_bus:
            return

        if isinstance(event, EventType):
            event = Event(type=event)
        message_data = event.model_dump_json()
        self.event_bus.send_string(f"{topic} {message_data}")

