import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any

import zmq
from lib_defs.type_defs.base_events import BaseEvent, BaseEventTopic, BaseEventType

from lib_utils.messages.consumer_task import consumer_task

logger = logging.getLogger(__name__)

type Callback[EventClass: BaseEvent] = Callable[[EventClass], Any]

class BaseEventBus[
    EventClass: BaseEvent,
    EventTopicClass: BaseEventTopic,
    EventTypeClass: BaseEventType,
]:
    """
    Generic ZMQ event bus for consuming pub-sub events.
    """
    def __init__(
        self,
        event_class: type[EventClass],
        topic: EventTopicClass,
        consumer_name: str = "event_bus_consumer",
        subscriber_addrs: list[str] | None = None,
        publisher_addr: str | None = None,
    ):
        self._event_class = event_class
        self.queue: Queue[EventClass] = Queue()
        self.zmq_context = zmq.Context()
        self.subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.publisher_socket = None
        self.stop_event = threading.Event()
        self.consumer_thread = threading.Thread(
            target=consumer_task,
            args=(self.subscriber_socket, self.queue, self.stop_event),
            kwargs={
                "event_class": event_class,
                "topic": topic,
            },
            daemon=True,
            name=consumer_name,
        )

        if subscriber_addrs is not None:
            for sub_socket in subscriber_addrs:
                self.subscriber_socket.bind(sub_socket)

        if publisher_addr is not None:
            self.publisher_socket = self.zmq_context.socket(zmq.PUB)
            self.publisher_socket.connect(publisher_addr)


        self._callbacks: dict[EventTypeClass, list[Callback]] = defaultdict(list)

    # ------------------------
    # --- Consumer Methods ---
    # ------------------------
    def start(self):
        self.consumer_thread.start()

    def recv_event(self) -> EventClass | None:
        if self.stop_event.is_set():
            return None

        try:
            event = self.queue.get(timeout=0.5)
        except Empty:
            return None

        self.queue.task_done()
        return event

    def listen_forever(self):
        while True:
            event: EventClass | None = self.recv_event()
            if event is None:
                continue

            for cb in self._callbacks[event.type]:
                cb(event)

            yield event

    # ------------------------
    # --- Producer Methods ---
    # ------------------------
    def emit_event(self, event: EventClass | EventTypeClass, topic: EventTopicClass):
        if self.publisher_socket is None:
            logger.error("Event Bus is trying to emit event without having publisher socket")
            return

        if not isinstance(event, self._event_class):
            # Then event is instance of EventTypeClass
            event = self._event_class(type=event)

        message_data = event.model_dump_json()
        self.publisher_socket.send_string(f"{topic} {message_data}")

    def attach_callback(self, event_type: EventTypeClass, cb: Callback):
        self._callbacks[event_type].append(cb)


    def attach_callbacks(self, cbs: dict[EventTypeClass, Callback]):
        for event_type, cb in cbs.items():
            self.attach_callback(event_type, cb)


    # ----------------------
    # --- Common Methods ---
    # ----------------------
    def cleanup(self):
        logger.info("Stopping event bus...")
        self.stop_event.set()
        # Drain the queue (Python <3.13 compatibility)
        empty_raised = False
        while not empty_raised:
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
                logger.debug("Draining queue...")
            except Empty:
                empty_raised = True
                logger.debug("Queue is empty, stopping drain loop")

        logger.debug("Joining queue...")
        self.queue.join()

        logger.debug("Joining consumer thread...")
        self.consumer_thread.join(timeout=5)

        logger.debug("Closing subscriber socket...")
        self.subscriber_socket.close()

        logger.debug("Closing publisher socket...")
        if self.publisher_socket is not None:
            self.publisher_socket.close()

        logger.debug("Terminating ZMQ context...")
        self.zmq_context.term()

        logger.info("Cleaned up event bus")
