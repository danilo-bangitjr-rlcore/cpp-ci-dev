from coredinator.event_bus.proxy import EventBusProxy
from coredinator.logging_config import get_logger

logger = get_logger(__name__)

class EventBusManager:
    def __init__(
        self,
        xsub_addr: str = "tcp://*:5559",
        xpub_addr: str = "tcp://*:5560",
    ):
        self.xsub_addr = xsub_addr
        self.xpub_addr = xpub_addr
        self.proxy = EventBusProxy(xsub_addr=xsub_addr, xpub_addr=xpub_addr)

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
            "xsub_addr": self.xsub_addr,
            "xpub_addr": self.xpub_addr,
            "publisher_endpoint": self.xsub_addr.replace("*", "localhost"),
            "subscriber_endpoint": self.xpub_addr.replace("*", "localhost"),
        }
