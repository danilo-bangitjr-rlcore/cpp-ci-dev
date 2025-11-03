import logging
import threading
from queue import Queue

import zmq
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def consumer_task(
    sub_socket: zmq.Socket,
    queue: Queue,
    stop_event: threading.Event,
    *,
    event_class: BaseModel,
    topic: str,
):
    """
    Generic ZMQ thread worker that consumes ZMQ messages
    and pushes messages into a python queue.

    event_class: class with model_validate_json method
    topic: topic string to subscribe to
    """
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    while not stop_event.is_set():
        try:
            poll_resp = sub_socket.poll(timeout=1000)
            if poll_resp == 0:
                continue
            raw_payload = sub_socket.recv()
            _raw_topic, raw_event = raw_payload.split(b" ", 1)
            event = event_class.model_validate_json(raw_event)
            logger.debug(f"Adding to queue Event: {event}")
            queue.put(event)
        except zmq.ZMQError as e:
            if isinstance(e, zmq.error.Again):
                logger.debug("ZMQ Again error, continuing to poll")
            elif e.errno == zmq.ETERM:
                logger.debug("ZMQ ETERM error, stopping consumer task")
                break
            else:
                logger.debug(f"ZMQ error occurred: {e}")
                raise
