import logging
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

import zmq
from pydantic import BaseModel

from lib_events.protocol.message_protocol import MessageType, build_message, parse_message

if TYPE_CHECKING:
    from typing import Protocol

    class EventProtocol(Protocol):
        type: Any
        def model_dump_json(self) -> str: ...
        @classmethod
        def model_validate_json(cls, data: str) -> "EventProtocol": ...

logger = logging.getLogger(__name__)

type Callback = Callable[[Any], Any]


class EventBusClient[EventClass: BaseModel, EventTypeClass: Enum, EventTopicClass: Enum]:
    def __init__(
        self,
        event_class: type[EventClass],
        host: str = "localhost",
        port: int = 5580,
        service_id: str | None = None,
        max_reconnect_attempts: int = -1,
        reconnect_interval: float = 1.0,
        reconnect_backoff_multiplier: float = 2.0,
        reconnect_max_interval: float = 60.0,
    ):
        self._event_class = event_class
        self.host = host
        self.port = port
        self.endpoint = f"tcp://{host}:{port}"
        self.service_id = service_id or f"client-{uuid.uuid4().hex[:8]}"

        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_interval = reconnect_interval
        self.reconnect_backoff_multiplier = reconnect_backoff_multiplier
        self.reconnect_max_interval = reconnect_max_interval

        self.context: zmq.Context | None = None
        self.dealer_socket: zmq.Socket | None = None

        self.queue: Queue[EventClass] = Queue()
        self.stop_event = threading.Event()
        self.consumer_thread: threading.Thread | None = None
        self._callbacks: dict[EventTypeClass, list[Callback]] = defaultdict(list)

        self._connected = False
        self._reconnect_attempts = 0
        self._subscribed_topics: list[EventTopicClass] = []
        self._reconnect_lock = threading.Lock()

    # ============================================================
    # Lifecycle Management
    # ============================================================

    def connect(self):
        if self._connected:
            logger.warning("Event bus client already connected")
            return

        self._setup_sockets()
        self._register_with_broker()
        self._resubscribe_topics()
        self._connected = True
        self._reconnect_attempts = 0
        logger.info(f"Event bus client connected - service_id: {self.service_id}, endpoint: {self.endpoint}")

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

        self._close_sockets()

        self._connected = False
        logger.info("Event bus client closed")

    def is_connected(self):
        return self._connected

    # ============================================================
    # Publishing
    # ============================================================

    def emit_event(self, event: EventClass | EventTypeClass, topic: EventTopicClass):
        if not self._connected or self.dealer_socket is None:
            logger.warning("Cannot publish - event bus client not connected")
            return

        if not isinstance(event, self._event_class):
            event = self._event_class(type=event)

        message_data = event.model_dump_json()
        topic_str = topic.name

        publish_msg = build_message(
            destination=topic_str,
            msg_type=MessageType.PUBLISH,
            correlation_id="",
            payload=message_data.encode(),
        )
        self.dealer_socket.send_multipart(publish_msg)
        event_type = getattr(event, "type", "unknown")
        logger.debug(f"Published event {event_type} to topic '{topic_str}'")

    # ============================================================
    # Subscribing
    # ============================================================

    def subscribe(self, topic: EventTopicClass):
        if self.dealer_socket is None:
            logger.warning("Cannot subscribe - dealer socket not initialized")
            return

        if topic not in self._subscribed_topics:
            self._subscribed_topics.append(topic)

        if self._connected:
            self._send_subscribe(topic)

        logger.info(f"Subscribed to topic: {topic.name}")

    def start_consumer(self):
        if self.consumer_thread is not None:
            logger.warning("Consumer thread already started")
            return

        if self.dealer_socket is None:
            logger.error("Cannot start consumer - dealer socket not initialized")
            return

        self.stop_event.clear()
        self.consumer_thread = threading.Thread(
            target=self._consume_messages,
            daemon=True,
            name="event_bus_client_consumer",
        )
        self.consumer_thread.start()
        logger.info("Event bus client consumer started")

    def attach_callback(self, event_type: EventTypeClass, cb: Callback):
        self._callbacks[event_type].append(cb)
        logger.debug(f"Attached callback for event type: {event_type}")

    def attach_callbacks(self, cbs: dict[EventTypeClass, Callback]):
        for event_type, cb in cbs.items():
            self.attach_callback(event_type, cb)

    def recv_event(self, timeout: float = 0.5) -> EventClass | None:
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

    def _close_sockets(self):
        if self.dealer_socket:
            self.dealer_socket.setsockopt(zmq.LINGER, 0)
            self.dealer_socket.close()
            self.dealer_socket = None

        if self.context:
            self.context.destroy(linger=0)
            self.context = None

    def _setup_sockets(self):
        self.context = zmq.Context()

        self.dealer_socket = self.context.socket(zmq.DEALER)
        assert self.dealer_socket is not None
        self.dealer_socket.setsockopt(zmq.LINGER, 1000)
        self.dealer_socket.setsockopt(zmq.SNDHWM, 1000)
        self.dealer_socket.connect(self.endpoint)

        # Brief sleep to allow socket connection to establish
        time.sleep(0.1)
        logger.debug(f"Event bus client DEALER socket connected to {self.endpoint}")

    def _register_with_broker(self):
        if self.dealer_socket is None:
            return

        register_msg = build_message(
            destination=self.service_id,
            msg_type=MessageType.REGISTER,
            correlation_id="",
            payload=b"{}",
        )
        self.dealer_socket.send_multipart(register_msg)
        logger.debug(f"Registered with broker as: {self.service_id}")

    def _send_subscribe(self, topic: EventTopicClass):
        if self.dealer_socket is None:
            return

        subscribe_msg = build_message(
            destination=topic.name,
            msg_type=MessageType.SUBSCRIBE,
            correlation_id="",
            payload=b"{}",
        )
        self.dealer_socket.send_multipart(subscribe_msg)
        logger.debug(f"Sent SUBSCRIBE for topic: {topic.name}")

    def _resubscribe_topics(self):
        for topic in self._subscribed_topics:
            self._send_subscribe(topic)

    def _reconnect(self) -> bool:
        with self._reconnect_lock:
            if self.stop_event.is_set():
                logger.debug("Stop event set, aborting reconnection")
                return False

            if self.max_reconnect_attempts >= 0 and self._reconnect_attempts >= self.max_reconnect_attempts:
                logger.error(
                    f"Max reconnection attempts ({self.max_reconnect_attempts}) reached, giving up",
                )
                return False

            self._reconnect_attempts += 1

            current_interval = min(
                self.reconnect_interval * (self.reconnect_backoff_multiplier ** (self._reconnect_attempts - 1)),
                self.reconnect_max_interval,
            )

            logger.info(
                f"Attempting to reconnect (attempt {self._reconnect_attempts}) in {current_interval:.2f}s",
            )

            sleep_start = time.time()
            while time.time() - sleep_start < current_interval:
                if self.stop_event.is_set():
                    logger.debug("Stop event set during reconnect sleep, aborting")
                    return False
                time.sleep(0.1)

            try:
                self._close_sockets()
                self._connected = False

                self._setup_sockets()
                self._register_with_broker()
                self._resubscribe_topics()

                self._connected = True
                self._reconnect_attempts = 0
                logger.info("Successfully reconnected to event bus")
                return True

            except Exception as e:
                logger.error(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")
                return False

    def _consume_messages(self):
        assert self.dealer_socket is not None

        poller = zmq.Poller()
        poller.register(self.dealer_socket, zmq.POLLIN)

        while not self.stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=500))
            except zmq.ZMQError as e:
                if self.stop_event.is_set():
                    break

                logger.warning(f"ZMQ polling error: {e}, attempting reconnection")
                if self._reconnect():
                    poller = zmq.Poller()
                    poller.register(self.dealer_socket, zmq.POLLIN)
                    continue

                logger.error("Failed to reconnect, stopping consumer")
                break

            if self.dealer_socket not in socks:
                continue

            try:
                message_frames = self.dealer_socket.recv_multipart(zmq.NOBLOCK)

                try:
                    parsed = parse_message(message_frames)
                    if parsed.is_none():
                        logger.warning("Failed to parse message")
                        continue

                    msg = parsed.expect()
                except Exception as e:
                    logger.warning(f"Failed to parse message: {e}")
                    continue

                if msg.msg_type != MessageType.PUBLISH:
                    logger.debug(f"Ignoring non-PUBLISH message type: {msg.msg_type}")
                    continue

                payload = msg.payload.decode()
                event = self._event_class.model_validate_json(payload)
                self.queue.put(event)

                event_type = getattr(event, "type", None)
                if event_type is not None:
                    for cb in self._callbacks[event_type]:
                        try:
                            cb(event)
                        except Exception as e:
                            logger.error(f"Callback error for event {event_type}: {e}", exc_info=True)

            except zmq.ZMQError as e:
                logger.warning(f"Error receiving message: {e}, attempting reconnection")
                if self._reconnect():
                    poller = zmq.Poller()
                    poller.register(self.dealer_socket, zmq.POLLIN)
                    continue

                logger.error("Failed to reconnect, stopping consumer")
                break
            except Exception as e:
                logger.error(f"Error consuming message: {e}", exc_info=True)

        logger.info("Consumer thread stopped")
