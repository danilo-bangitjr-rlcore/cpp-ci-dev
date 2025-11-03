from lib_defs.type_defs.base_events import Event, EventTopic, EventType
from lib_events.client.event_bus_client import EventBusClient as _EventBusClient


class RLEventBusClient(_EventBusClient[Event, EventType, EventTopic]):
    def __init__(
        self,
        host: str = "localhost",
        pub_port: int = 5559,
        sub_port: int = 5560,
    ):
        super().__init__(
            event_class=Event,
            host=host,
            pub_port=pub_port,
            sub_port=sub_port,
        )


EventBusClient = RLEventBusClient
