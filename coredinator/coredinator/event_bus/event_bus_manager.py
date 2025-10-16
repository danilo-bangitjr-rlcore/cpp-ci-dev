from coredinator.event_bus.proxy import EventBusProxy
from coredinator.logging_config import get_logger

logger = get_logger(__name__)

class EventBusManager:
    def __init__(
        self,
        host: str = "*",
        pub_port: int = 5559,
        sub_port: int = 5560,
    ):
        self.host = host
        self.pub_port = pub_port
        self.sub_port = sub_port
        # ZMQ proxy architecture (counterintuitive naming):
        # - XSUB socket: where publishers CONNECT (binds to pub_port)
        # - XPUB socket: where subscribers CONNECT (binds to sub_port)
        self.xsub_addr = f"tcp://{host}:{pub_port}"
        self.xpub_addr = f"tcp://{host}:{sub_port}"
        self.proxy = EventBusProxy(xsub_addr=self.xsub_addr, xpub_addr=self.xpub_addr)

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
