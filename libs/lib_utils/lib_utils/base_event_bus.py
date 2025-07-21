import logging
import threading
from queue import Empty, Queue
from typing import Generic, TypeVar

import zmq
from lib_defs.type_defs.base_events import BaseEvent

from lib_utils.consumer_task import consumer_task

logger = logging.getLogger(__name__)

EventClass = TypeVar('EventClass', bound='BaseEvent')

class BaseEventBus(Generic[EventClass]): # noqa: UP046
    """
    Generic ZMQ event bus for consuming pub-sub events.
    """
    def __init__(
        self,
        event_class: type[EventClass],
        topic: str,
        consumer_name: str = "event_bus_consumer",
        subscriber_sockets: list[str] | None = None,
    ):
        self.queue: Queue[EventClass] = Queue()
        self.zmq_context = zmq.Context()
        self.subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.publisher_socket = self.zmq_context.socket(zmq.PUB)
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

        if isinstance(subscriber_sockets, list):
            for sub_socket in subscriber_sockets:
                self.subscriber_socket.bind(sub_socket)

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
        self.publisher_socket.close()

        logger.debug("Terminating ZMQ context...")
        self.zmq_context.term()

        logger.info("Cleaned up event bus")
