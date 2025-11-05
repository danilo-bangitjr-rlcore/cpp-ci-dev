from lib_defs.type_defs.base_events import Event, EventTopic, EventType
from lib_events.client.event_bus_client import EventBusClient as _EventBusClient


class RLEventBusClient(_EventBusClient[Event, EventType, EventTopic]):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5580,
    ):
        super().__init__(
            event_class=Event,
            host=host,
            port=port,
        )


EventBusClient = RLEventBusClient
