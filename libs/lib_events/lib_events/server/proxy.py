import logging
import threading
from collections import defaultdict

import zmq

from lib_events.protocol.message_protocol import MessageType, ParsedMessage, parse_message

logger = logging.getLogger(__name__)


class EventBusProxy:
    def __init__(
        self,
        router_addr: str = "tcp://*:5580",
    ):
        self.router_addr = router_addr
        self.context: zmq.Context | None = None
        self.router_socket: zmq.Socket | None = None
        self._proxy_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._service_registry: dict[str, bytes] = {}
        self._topic_subscriptions: dict[str, set[bytes]] = defaultdict(set)
        self._pending_requests: dict[str, bytes] = {}
        self._lock = threading.Lock()

    # ---------------
    # -- Lifecycle --
    # ---------------

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
        logger.info(f"Event bus proxy started - router: {self.router_addr}")

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

    # ----------------------
    # -- Message Handling --
    # ----------------------

    def _message_loop(self):
        assert self.router_socket is not None

        poller = zmq.Poller()
        poller.register(self.router_socket, zmq.POLLIN)

        while not self._stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=500))
            except zmq.ZMQError as e:
                if self._stop_event.is_set():
                    break

                logger.error(f"ZMQ polling error: {e}")
                continue

            if self.router_socket in socks:
                self._handle_message()

    def _handle_message(self):
        assert self.router_socket is not None

        try:
            frames = self.router_socket.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return

        if len(frames) < 5:
            logger.warning(f"Received malformed message with {len(frames)} frames")
            return

        zmq_identity = frames[0]
        message_frames = frames[1:]

        parsed = parse_message(message_frames)
        if parsed.is_none():
            logger.error("Failed to parse message")
            return

        msg = parsed.expect()

        if msg.msg_type == MessageType.REGISTER:
            self._handle_register(zmq_identity, msg.destination)
        elif msg.msg_type == MessageType.SUBSCRIBE:
            self._handle_subscribe(zmq_identity, msg.destination)
        elif msg.msg_type == MessageType.PUBLISH:
            self._handle_publish(msg)
        elif msg.msg_type == MessageType.REQUEST:
            self._handle_request(zmq_identity, msg)
        elif msg.msg_type == MessageType.REPLY:
            self._handle_reply(msg)
        else:
            logger.warning(f"Unknown message type: {msg.msg_type}")

    # ---------------------------
    # -- Message Type Handlers --
    # ---------------------------

    def _handle_register(self, zmq_identity: bytes, service_id: str):
        with self._lock:
            if service_id in self._service_registry:
                logger.warning(
                    f"Service '{service_id}' already registered, overwriting",
                )
            self._service_registry[service_id] = zmq_identity
            logger.info(f"Registered service: {service_id}")

    def _handle_subscribe(self, zmq_identity: bytes, topic: str):
        with self._lock:
            self._topic_subscriptions[topic].add(zmq_identity)
            logger.debug(
                f"Subscription added: topic={topic}, "
                f"subscribers={len(self._topic_subscriptions[topic])}",
            )

    def _handle_publish(self, msg: ParsedMessage):
        assert self.router_socket is not None

        with self._lock:
            subscribers = self._topic_subscriptions.get(msg.destination, set())

        if not subscribers:
            logger.debug(f"No subscribers for topic: {msg.destination}")
            return

        logger.debug(f"Publishing to {len(subscribers)} subscribers on topic: {msg.destination}")

        for subscriber_id in subscribers:
            _send_frames(self.router_socket, subscriber_id, msg.frames)

    def _handle_request(
        self,
        requester_id: bytes,
        msg: ParsedMessage,
    ):
        assert self.router_socket is not None

        with self._lock:
            service_id = self._service_registry.get(msg.destination)
            self._pending_requests[msg.correlation_id] = requester_id

        if service_id is None:
            logger.warning(f"Request to unregistered service: {msg.destination}")
            self._send_error_reply(requester_id, msg)
            with self._lock:
                self._pending_requests.pop(msg.correlation_id, None)
            return

        _send_frames(self.router_socket, service_id, msg.frames)
        logger.debug(f"Routed REQUEST to service: {msg.destination} (correlation_id: {msg.correlation_id})")

    def _handle_reply(self, msg: ParsedMessage):
        assert self.router_socket is not None

        with self._lock:
            requester_id = self._pending_requests.pop(msg.correlation_id, None)

        if requester_id is None:
            logger.warning(f"Reply with unknown correlation_id: {msg.correlation_id}")
            return

        _send_frames(self.router_socket, requester_id, msg.frames)
        logger.debug(f"Routed REPLY for correlation_id: {msg.correlation_id}")

    def _send_error_reply(self, requester_id: bytes, msg: ParsedMessage):
        assert self.router_socket is not None

        error_payload = b'{"error": "Service not available"}'
        error_frames = [
            b"",
            MessageType.REPLY.value.encode("utf-8"),
            msg.correlation_id.encode("utf-8"),
            error_payload,
        ]

        _send_frames(self.router_socket, requester_id, error_frames)
        logger.debug(f"Sent error reply for correlation_id: {msg.correlation_id}")

    # -----------------------
    # -- Socket Management --
    # -----------------------

    def _run_proxy(self):
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)

        try:
            assert self.router_socket is not None

            self.router_socket.bind(self.router_addr)
            logger.debug(f"ZMQ router socket bound - router: {self.router_addr}")

            self._message_loop()

        except Exception as e:
            logger.error(f"Event bus proxy error: {e}", exc_info=True)
        finally:
            self._cleanup_sockets()

    def _cleanup_sockets(self):
        if self.router_socket:
            self.router_socket.setsockopt(zmq.LINGER, 0)
            self.router_socket.close()
            self.router_socket = None

        if self.context:
            self.context.destroy(linger=0)
            self.context = None

        with self._lock:
            self._service_registry.clear()
            self._topic_subscriptions.clear()
            self._pending_requests.clear()

        logger.debug("ZMQ router socket cleaned up")

    # -------------------
    # -- Query Methods --
    # -------------------

    def get_service_count(self):
        with self._lock:
            return len(self._service_registry)

    def get_topic_subscriber_count(self, topic: str):
        with self._lock:
            return len(self._topic_subscriptions.get(topic, set()))


# ---------------
# -- Utilities --
# ---------------

def _send_frames(socket: zmq.Socket, identity: bytes, frames: list[bytes]):
    try:
        socket.send_multipart(
            [identity, *frames],
            zmq.NOBLOCK,
        )
    except zmq.Again:
        logger.warning("Failed to send message (queue full)")
    except zmq.ZMQError as e:
        logger.error(f"Error sending message: {e}")
