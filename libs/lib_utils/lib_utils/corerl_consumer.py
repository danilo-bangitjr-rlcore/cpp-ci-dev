import logging
import threading
from queue import Queue
from typing import Any, Callable, Type

import zmq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def consumer_task(
    sub_socket: zmq.Socket,
    queue: Queue,
    stop_event: threading.Event,
    *,
    event_class: Type[Any],
    topic: str,
    should_toggle_logging: bool = False
):
    """
    Generic ZMQ consumer task for use with any event type and topic.
    event_class: class with model_validate_json method
    topic: topic string to subscribe to
    should_toggle_logging: whether to toggle event logging
    """
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    while not stop_event.is_set():
        try:
            poll_resp = sub_socket.poll(timeout=1000)
            if poll_resp == 0:
                continue
            raw_payload = sub_socket.recv()
            raw_topic, raw_event = raw_payload.split(b" ", 1)
            event = event_class.model_validate_json(raw_event)
            if should_toggle_logging:
                toggle_event_logging()
            logger.debug(f"Adding to queue Event: {event}")
            queue.put(event)
        except zmq.ZMQError as e:
            if isinstance(e, zmq.error.Again):
                pass
            elif e.errno == zmq.ETERM:
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
