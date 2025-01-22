import logging
import threading
from queue import Queue

import zmq

from corerl.messages.events import Event, EventTopic


def consumer_task(sub_socket: zmq.Socket, queue: Queue, stop_event: threading.Event):
    """
    Thread worker that consumes ZMQ messages and pushes messages into a python queue.
    """
    _logger = logging.getLogger(__name__)

    topic = EventTopic.corerl

    sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    while not stop_event.is_set():
        try:
            poll_resp = sub_socket.poll(timeout=1000)
            if poll_resp == 0:
                continue
            raw_payload = sub_socket.recv()
            raw_topic, raw_event = raw_payload.split(b" ", 1)
            event = Event.model_validate_json(raw_event)
            queue.put(event)
        except zmq.ZMQError as e:
            if isinstance(e, zmq.error.Again):
                # temporarily unavailable, retry
                pass
            elif e.errno == zmq.ETERM:
                # exit, break from loop
                break
            else:
                raise
