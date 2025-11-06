import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any

import zmq
from lib_defs.type_defs.base_events import Event, EventTopic, EventType

from lib_events.protocol.message_protocol import MessageType, build_message, parse_message

logger = logging.getLogger(__name__)

type Callback = Callable[[Any], Any]


class EventBusClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5580,
        service_id: str | None = None,
        max_reconnect_attempts: int = -1,
        reconnect_interval: float = 1.0,
        reconnect_backoff_multiplier: float = 2.0,
        reconnect_max_interval: float = 60.0,
    ):
        self.port = port
        self.endpoint = f"tcp://{host}:{port}"
        self.service_id = service_id or f"client-{uuid.uuid4().hex[:8]}"

        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_interval = reconnect_interval
        self.reconnect_backoff_multiplier = reconnect_backoff_multiplier
        self.reconnect_max_interval = reconnect_max_interval

        self.context: zmq.Context | None = None
        self.dealer_socket: zmq.Socket | None = None

        self.queue: Queue[Event] = Queue()
        self.stop_event = threading.Event()
        self.consumer_thread: threading.Thread | None = None
        self._callbacks: dict[EventType, list[Callback]] = defaultdict(list)

        self._connected = False
        self._reconnect_attempts = 0
        self._subscribed_topics: list[EventTopic] = []
        self._reconnect_lock = threading.Lock()

        self._request_handler: Callable[[bytes], bytes] | None = None
        self._pending_replies: dict[str, bytes] = {}
        self._pending_replies_lock = threading.Lock()
        self._socket_send_lock = threading.Lock()

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

        self.stop_consumer()

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

    def emit_event(self, event: Event | EventType, topic: EventTopic):
        if not self._connected or self.dealer_socket is None:
            logger.warning("Cannot publish - event bus client not connected")
            return

        if not isinstance(event, Event):
            event = Event(type=event)

        message_data = event.model_dump_json()
        topic_str = topic.name

        publish_msg = build_message(
            destination=topic_str,
            msg_type=MessageType.PUBLISH,
            correlation_id="",
            payload=message_data.encode(),
        )
        self._send_message(publish_msg)

        logger.debug(f"Published event {event.type} to topic '{topic_str}'")

    # ============================================================
    # Request/Reply
    # ============================================================

    def send_request(
        self,
        service_id: str,
        payload: bytes,
        timeout: float = 5.0,
    ) -> bytes:
        return asyncio.run(self.send_request_async(service_id, payload, timeout))

    async def send_request_async(
        self,
        service_id: str,
        payload: bytes,
        timeout: float = 5.0,
    ) -> bytes:
        if not self._connected or self.dealer_socket is None:
            raise RuntimeError("Event bus client not connected")

        if self.consumer_thread is None or not self.consumer_thread.is_alive():
            raise RuntimeError("Consumer thread must be running to send requests")

        correlation_id = uuid.uuid4().hex

        request_msg = build_message(
            destination=service_id,
            msg_type=MessageType.REQUEST,
            correlation_id=correlation_id,
            payload=payload,
        )
        self._send_message(request_msg)
        logger.debug(f"Sent REQUEST to {service_id} with correlation_id: {correlation_id}")

        return await self._wait_for_reply(correlation_id, timeout)

    def register_request_handler(self, handler: Callable[[bytes], bytes]):
        self._request_handler = handler
        logger.info("Registered request handler")

    async def _wait_for_reply(self, correlation_id: str, timeout: float) -> bytes:
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed >= timeout:
                with self._pending_replies_lock:
                    self._pending_replies.pop(correlation_id, None)
                raise TimeoutError(f"Request timed out after {timeout}s")

            with self._pending_replies_lock:
                if correlation_id in self._pending_replies:
                    payload_bytes = self._pending_replies.pop(correlation_id)
                    logger.debug(f"Received REPLY with correlation_id: {correlation_id}")
                    return payload_bytes

            await asyncio.sleep(0.01)

    def _send_message(self, msg: list[bytes]):
        if self.dealer_socket is None:
            logger.error("Cannot send message - dealer socket not initialized")
            return
        with self._socket_send_lock:
            self.dealer_socket.send_multipart(msg)

    # ============================================================
    # Subscribing
    # ============================================================

    def subscribe(self, topic: EventTopic):
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

    def stop_consumer(self):
        if self.consumer_thread is None:
            logger.debug("Consumer thread not started, nothing to stop")
            return

        logger.info("Stopping consumer thread")
        self.stop_event.set()

        if self.consumer_thread is not None:
            self.consumer_thread.join(timeout=2)
            if self.consumer_thread.is_alive():
                logger.warning("Consumer thread did not terminate within timeout")
            else:
                logger.info("Consumer thread stopped")

        self.consumer_thread = None

    def attach_callback(self, event_type: EventType, cb: Callback):
        self._callbacks[event_type].append(cb)
        logger.debug(f"Attached callback for event type: {event_type}")

    def attach_callbacks(self, cbs: dict[EventType, Callback]):
        for event_type, cb in cbs.items():
            self.attach_callback(event_type, cb)

    def recv_event(self, timeout: float = 0.5) -> Event | None:
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
        register_msg = build_message(
            destination=self.service_id,
            msg_type=MessageType.REGISTER,
            correlation_id="",
            payload=b"{}",
        )
        self._send_message(register_msg)
        logger.debug(f"Registered with broker as: {self.service_id}")

    def _send_subscribe(self, topic: EventTopic):
        subscribe_msg = build_message(
            destination=topic.name,
            msg_type=MessageType.SUBSCRIBE,
            correlation_id="",
            payload=b"{}",
        )
        self._send_message(subscribe_msg)
        logger.debug(f"Sent SUBSCRIBE for topic: {topic.name}")

    def _resubscribe_topics(self):
        for topic in self._subscribed_topics:
            self._send_subscribe(topic)

    def _handle_publish_message(self, payload: bytes):
        event = Event.model_validate_json(payload.decode())
        self.queue.put(event)

        for cb in self._callbacks[event.type]:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Callback error for event {event.type}: {e}", exc_info=True)

    def _handle_reply_message(self, correlation_id: str, payload: bytes):
        with self._pending_replies_lock:
            self._pending_replies[correlation_id] = payload
        logger.debug(f"Stored REPLY with correlation_id: {correlation_id}")

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

            if not _interruptible_sleep(self.stop_event, current_interval):
                logger.debug("Stop event set during reconnect sleep, aborting")
                return False

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

                parsed = parse_message(message_frames)
                if parsed.is_none():
                    logger.warning("Failed to parse message")
                    continue

                msg = parsed.expect()

                if msg.msg_type == MessageType.PUBLISH:
                    self._handle_publish_message(msg.payload)
                elif msg.msg_type == MessageType.REQUEST:
                    self._handle_incoming_request(msg.correlation_id, msg.payload)
                elif msg.msg_type == MessageType.REPLY:
                    self._handle_reply_message(msg.correlation_id, msg.payload)
                else:
                    logger.debug(f"Ignoring message type: {msg.msg_type}")

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

    def _handle_incoming_request(self, correlation_id: str, payload: bytes):
        if self._request_handler is None:
            logger.warning(f"Received REQUEST but no handler registered (correlation_id: {correlation_id})")
            error_payload = b'{"error": "No request handler registered"}'
            self._send_reply(correlation_id, error_payload)
            return

        try:
            response_payload = self._request_handler(payload)
            self._send_reply(correlation_id, response_payload)
            logger.debug(f"Handled REQUEST with correlation_id: {correlation_id}")
        except Exception as e:
            logger.error(f"Request handler error: {e}", exc_info=True)
            error_payload = f'{{"error": "{e!s}"}}'.encode()
            self._send_reply(correlation_id, error_payload)

    def _send_reply(self, correlation_id: str, payload: bytes):
        reply_msg = build_message(
            destination="",
            msg_type=MessageType.REPLY,
            correlation_id=correlation_id,
            payload=payload,
        )
        self._send_message(reply_msg)
        logger.debug(f"Sent REPLY with correlation_id: {correlation_id}")


def _interruptible_sleep(stop_event: threading.Event, duration: float) -> bool:
    sleep_start = time.time()
    while time.time() - sleep_start < duration:
        if stop_event.is_set():
            return False
        time.sleep(0.1)
    return True
