import logging
import threading
from queue import Queue

from coreio.utils.zmq_communication import IOEvent, IOEventTopic

import zmq

logger = logging.getLogger(__name__)

def coreio_consumer_task(sub_socket: zmq.Socket, queue: Queue, stop_event: threading.Event):
    topic = IOEventTopic.coreio

    sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    while not stop_event.is_set():
        try:
            poll_resp = sub_socket.poll(timeout=1000)
            if poll_resp == 0:
                    continue
            raw_payload = sub_socket.recv()
            raw_topic, raw_event = raw_payload.split(b" ", 1)
            event = IOEvent.model_validate_json(raw_event)
            logger.debug(f"Adding to queue IO Event: {event}")
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

