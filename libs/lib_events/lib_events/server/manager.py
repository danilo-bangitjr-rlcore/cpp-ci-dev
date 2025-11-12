import logging

from lib_events.server.proxy import EventBusProxy

logger = logging.getLogger(__name__)

class EventBusManager:
    def __init__(self, host: str = "*", port: int = 5580):
        self.host = host
        self.port = port
        self.router_addr = f"tcp://{host}:{port}"
        self.proxy = EventBusProxy(router_addr=self.router_addr)

    def start(self):
        logger.info("Starting event bus manager")
        self.proxy.start()

    def stop(self):
        logger.info("Stopping event bus manager")
        self.proxy.stop()

    def is_healthy(self):
        return self.proxy.is_running()

    def get_config(self):
        return {
            "router_addr": self.router_addr,
            "endpoint": self.router_addr.replace("*", "localhost"),
        }

    def get_service_count(self):
        return self.proxy.get_service_count()

    def get_topic_subscriber_count(self, topic: str):
        return self.proxy.get_topic_subscriber_count(topic)
