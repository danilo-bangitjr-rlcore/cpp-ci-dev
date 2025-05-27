import logging
import threading
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any

import zmq
from pydantic import ValidationError

from coreio.config import CoreIOConfig
from coreio.utils.consumer_task import coreio_consumer_task
from coreio.utils.io_events import IOEvent

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
        self.subscriber_socket.bind(cfg.coreio_origin)

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
        self.zmq_context.term()

        logger.info("Cleaned up event bus")
