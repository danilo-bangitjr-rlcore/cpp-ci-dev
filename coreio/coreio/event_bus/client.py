from lib_events.client.event_bus_client import EventBusClient as _EventBusClient


class IOEventBusClient(_EventBusClient):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5580,
    ):
        super().__init__(
            host=host,
            port=port,
        )


EventBusClient = IOEventBusClient
