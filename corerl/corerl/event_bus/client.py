from lib_utils.messages.event_bus_client import EventBusClient as _EventBusClient

from corerl.messages.events import RLEvent, RLEventTopic, RLEventType


class RLEventBusClient(_EventBusClient[RLEvent, RLEventType, RLEventTopic]):
    def __init__(
        self,
        host: str = "localhost",
        pub_port: int = 5559,
        sub_port: int = 5560,
    ):
        super().__init__(
            event_class=RLEvent,
            host=host,
            pub_port=pub_port,
            sub_port=sub_port,
        )


EventBusClient = RLEventBusClient
