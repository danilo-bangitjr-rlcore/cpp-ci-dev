from coreio.config import CoreIOConfig
from coreio.utils.consumer_task import coreio_consumer_task
import logging

from collections import defaultdict
from collections.abc import Callable
from typing import Any
from pydantic import ValidationError

from queue import Queue, Empty
import threading
from coreio.utils.io_events import IOEvent, IOEventType, IOEventTopic

import zmq

logger = logging.getLogger(__name__)

Callback = Callable[[IOEvent], Any]

# ---------------------
# -- Utility methods --
# ---------------------
def maybe_parse_event(msg: str | bytes) -> IOEvent | None:
    try:
        return IOEvent.model_validate_json(msg)
    except ValidationError:
        logger.exception('Failed to parse message')
        return None


class ZMQ_Communication:
    def __init__(self, cfg: CoreIOConfig):
        self.queue = Queue()
        self.zmq_context = zmq.Context()

        self.subscriber_socket = self.zmq_context.socket(zmq.SUB)
        # self.publisher_socket = self.zmq_context.socket(zmq.PUB)

        self.coreio_stop_event = threading.Event()
        self.consumer_thread = threading.Thread(
            target=coreio_consumer_task,
            args=(
                self.subscriber_socket,
                self.queue,
                self.coreio_stop_event
            ),
            daemon=True,
            name="coreio_consumer"
        )

        self.subscriber_socket.bind(cfg.coreio_connection)
        # self.publisher_socket.bind(cfg.coreio_connection)

        # self._callbacks: dict[str, list[Callback]] = defaultdict(list)

    def start(self):
        self.consumer_thread.start()

    def recv_event(self) -> None | IOEvent:
        if self.coreio_stop_event.is_set():
            return None

        event = None

        try:
            event = self.queue.get(True, 0.5)
            return event
        except Empty:
            return None
        finally:
            if event:
                self.queue.task_done()
    
    # def listen_forever(self):
    #     while True:
    #         event = self.recv_event()
    #         if event is None:
    #             continue
    #
    #         for cb in self._callbacks[event.type]:
    #             cb(event)
    #
    #         yield event
    #
    #
    # def attach_callback(self, event_type: IOEventType, cb: Callback):
    #     self._callbacks[event_type].append(cb)
    #
    #
    # def attach_callbacks(self, cbs: dict[IOEventType, Callback]):
    #     for event_type, cb in cbs.items():
    #         self.attach_callback(event_type, cb)


    def cleanup(self):
        self.coreio_stop_event.set()

        # queue.shutdown introduced in Python 3.13, for now consume all items and then join
        empty_raised = False
        while not empty_raised:
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
            except Empty:
                empty_raised = True
        self.queue.join()

        self.consumer_thread.join()

        self.subscriber_socket.close()
        # self.publisher_socket.close()
        self.zmq_context.term()

        logger.info("Cleaned up event bus")
