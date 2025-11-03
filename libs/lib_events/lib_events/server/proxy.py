import logging
import threading

import zmq

logger = logging.getLogger(__name__)

class EventBusProxy:
    def __init__(
        self,
        xsub_addr: str = "tcp://*:5570",
        xpub_addr: str = "tcp://*:5571",
    ):
        self.xsub_addr = xsub_addr
        self.xpub_addr = xpub_addr
        self.context: zmq.Context | None = None
        self.xsub_socket: zmq.Socket | None = None
        self.xpub_socket: zmq.Socket | None = None
        self._proxy_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self):
        if self._running:
            logger.warning("Event bus proxy already running")
            return

        self._stop_event.clear()
        self._proxy_thread = threading.Thread(
            target=self._run_proxy,
            daemon=True,
            name="event_bus_proxy",
        )
        self._proxy_thread.start()
        self._running = True
        logger.info(
            f"Event bus proxy started - xsub: {self.xsub_addr}, xpub: {self.xpub_addr}",
        )

    def _forward_messages(self, xsub_socket: zmq.Socket, xpub_socket: zmq.Socket):
        poller = zmq.Poller()
        poller.register(xsub_socket, zmq.POLLIN)
        poller.register(xpub_socket, zmq.POLLIN)

        while not self._stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=500))
            except zmq.ZMQError as e:
                if self._stop_event.is_set():
                    break

                logger.error(f"ZMQ polling error: {e}")
                continue

            if xsub_socket in socks:
                message = xsub_socket.recv_multipart(zmq.NOBLOCK)
                xpub_socket.send_multipart(message)

            if xpub_socket in socks:
                message = xpub_socket.recv_multipart(zmq.NOBLOCK)
                xsub_socket.send_multipart(message)

    def _run_proxy(self):
        self.context = zmq.Context()
        self.xsub_socket = self.context.socket(zmq.XSUB)
        self.xpub_socket = self.context.socket(zmq.XPUB)

        try:
            assert self.xsub_socket is not None
            assert self.xpub_socket is not None

            self.xsub_socket.bind(self.xsub_addr)
            self.xpub_socket.bind(self.xpub_addr)
            logger.debug(
                f"ZMQ proxy sockets bound - xsub: {self.xsub_addr}, xpub: {self.xpub_addr}",
            )

            self._forward_messages(self.xsub_socket, self.xpub_socket)

        except Exception as e:
            logger.error(f"Event bus proxy error: {e}", exc_info=True)
        finally:
            self._cleanup_sockets()

    def _cleanup_sockets(self):
        if self.xsub_socket:
            self.xsub_socket.close()
            self.xsub_socket = None

        if self.xpub_socket:
            self.xpub_socket.close()
            self.xpub_socket = None

        if self.context:
            self.context.term()
            self.context = None

        logger.debug("ZMQ proxy sockets cleaned up")

    def stop(self):
        if not self._running:
            logger.debug("Event bus proxy not running, nothing to stop")
            return

        logger.info("Stopping event bus proxy")
        self._stop_event.set()

        if self._proxy_thread:
            self._proxy_thread.join(timeout=5)
            if self._proxy_thread.is_alive():
                logger.warning("Proxy thread did not terminate within timeout")

        self._running = False
        logger.info("Event bus proxy stopped")

    def is_running(self):
        return self._running and (self._proxy_thread is not None and self._proxy_thread.is_alive())
