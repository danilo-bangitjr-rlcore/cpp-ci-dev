import logging
import threading
from queue import Queue

import zmq

from corerl.messages.events import Event, EventTopic, EventType

logger = logging.getLogger(__name__)

def consumer_task(sub_socket: zmq.Socket, queue: Queue, stop_event: threading.Event):
    """
    Thread worker that consumes ZMQ messages and pushes messages into a python queue.
    """

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
            if event.type == EventType.toggle_event_logging:
                toggle_event_logging()
                continue
            logger.debug(f"Adding to queue Event: {event}")
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

def toggle_event_logging():
    interaction_logger = logging.getLogger("corerl.interaction.deployment_interaction")
    scheduler_logger = logging.getLogger("corerl.messages.scheduler")
    if logger.level != logging.DEBUG:
        # print event logs
        logger.setLevel(logging.DEBUG)
        scheduler_logger.setLevel(logging.DEBUG)
        interaction_logger.setLevel(logging.DEBUG)
    else:
        # do not print event logs
        logger.setLevel(logging.INFO)
        scheduler_logger.setLevel(logging.INFO)
        interaction_logger.setLevel(logging.INFO)
