import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any

import zmq

from corerl.messages.events import RLEvent, RLEventTopic, RLEventType

logger = logging.getLogger(__name__)

type Callback = Callable[[RLEvent], Any]


class EventBusClient:
    def __init__(
        self,
        host: str = "localhost",
        pub_port: int = 5559,
        sub_port: int = 5560,
    ):
        self.host = host
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.publisher_endpoint = f"tcp://{host}:{pub_port}"
        self.subscriber_endpoint = f"tcp://{host}:{sub_port}"

        self.context: zmq.Context | None = None
        self.publisher_socket: zmq.Socket | None = None
        self.subscriber_socket: zmq.Socket | None = None

        self.queue: Queue[RLEvent] = Queue()
        self.stop_event = threading.Event()
        self.consumer_thread: threading.Thread | None = None
        self._callbacks: dict[RLEventType, list[Callback]] = defaultdict(list)

        self._connected = False

    # ============================================================
    # Lifecycle Management
    # ============================================================

    def connect(self):
        if self._connected:
            logger.warning("Event bus client already connected")
            return

        self.context = zmq.Context()

        self.publisher_socket = self.context.socket(zmq.PUB)
        assert self.publisher_socket is not None
        self.publisher_socket.connect(self.publisher_endpoint)

        self.subscriber_socket = self.context.socket(zmq.SUB)
        assert self.subscriber_socket is not None
        self.subscriber_socket.connect(self.subscriber_endpoint)

        self._connected = True
        logger.info(
            f"Event bus client connected - pub: {self.publisher_endpoint}, sub: {self.subscriber_endpoint}",
        )

    def close(self):
        if not self._connected:
            logger.debug("Event bus client not connected, nothing to close")
            return

        logger.info("Closing event bus client")

        self.stop_event.set()

        if self.consumer_thread is not None:
            self.consumer_thread.join(timeout=5)
            if self.consumer_thread.is_alive():
                logger.warning("Consumer thread did not terminate within timeout")

        empty_raised = False
        while not empty_raised:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Empty:
                empty_raised = True

        self.queue.join()

        if self.subscriber_socket:
            self.subscriber_socket.close()
            self.subscriber_socket = None

        if self.publisher_socket:
            self.publisher_socket.close()
            self.publisher_socket = None

        if self.context:
            self.context.term()
            self.context = None

        self._connected = False
        logger.info("Event bus client closed")

    def is_connected(self):
        return self._connected

    # ============================================================
    # Publishing
    # ============================================================

    def emit_event(self, event: RLEvent | RLEventType, topic: RLEventTopic = RLEventTopic.corerl):
        if not self._connected or self.publisher_socket is None:
            logger.warning("Cannot publish - event bus client not connected")
            return

        if not isinstance(event, RLEvent):
            event = RLEvent(type=event)

        message_data = event.model_dump_json()
        topic_str = topic.name
        multipart_message = [topic_str.encode(), message_data.encode()]
        self.publisher_socket.send_multipart(multipart_message)
        logger.debug(f"Published event {event.type} to topic '{topic_str}'")

    # ============================================================
    # Subscribing
    # ============================================================

    def subscribe(self, topic: RLEventTopic):
        if self.subscriber_socket is None:
            logger.warning("Cannot subscribe - subscriber socket not initialized")
            return

        topic_bytes = topic.name.encode()
        self.subscriber_socket.setsockopt(zmq.SUBSCRIBE, topic_bytes)
        logger.info(f"Subscribed to topic: {topic.name}")

    def start_consumer(self):
        if self.consumer_thread is not None:
            logger.warning("Consumer thread already started")
            return

        if self.subscriber_socket is None:
            logger.error("Cannot start consumer - subscriber socket not initialized")
            return

        self.stop_event.clear()
        self.consumer_thread = threading.Thread(
            target=self._consume_messages,
            daemon=True,
            name="event_bus_client_consumer",
        )
        self.consumer_thread.start()
        logger.info("Event bus client consumer started")

    def attach_callback(self, event_type: RLEventType, cb: Callback):
        self._callbacks[event_type].append(cb)
        logger.debug(f"Attached callback for event type: {event_type}")

    def attach_callbacks(self, cbs: dict[RLEventType, Callback]):
        for event_type, cb in cbs.items():
            self.attach_callback(event_type, cb)

    def recv_event(self, timeout: float = 0.5) -> RLEvent | None:
        if self.stop_event.is_set():
            return None

        try:
            event = self.queue.get(timeout=timeout)
            self.queue.task_done()
            return event
        except Empty:
            return None

    def listen_forever(self):
        while not self.stop_event.is_set():
            event = self.recv_event()
            if event is None:
                continue

            yield event

    def _consume_messages(self):
        assert self.subscriber_socket is not None

        poller = zmq.Poller()
        poller.register(self.subscriber_socket, zmq.POLLIN)

        while not self.stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=500))
            except zmq.ZMQError:
                if self.stop_event.is_set():
                    break
                continue

            if self.subscriber_socket not in socks:
                continue

            try:
                message_parts = self.subscriber_socket.recv_multipart(zmq.NOBLOCK)
                if len(message_parts) != 2:
                    logger.warning(f"Invalid message format: expected 2 parts, got {len(message_parts)}")
                    continue

                payload = message_parts[1].decode()

                event = RLEvent.model_validate_json(payload)
                self.queue.put(event)

                for cb in self._callbacks[event.type]:
                    try:
                        cb(event)
                    except Exception as e:
                        logger.error(f"Callback error for event {event.type}: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"Error consuming message: {e}", exc_info=True)

        logger.info("Consumer thread stopped")
