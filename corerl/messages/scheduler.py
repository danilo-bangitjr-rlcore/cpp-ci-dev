import logging
import threading
import time

import zmq

from corerl.messages.events import Event, EventTopic, EventType
from corerl.messages.factory import EventBusConfig


def scheduler_task(event_bus_cfg: EventBusConfig, context: zmq.Context, stop_event: threading.Event):
    """
    Processs that emits ZMQ messages using our messages Event class.
    Stub for the CoreRL Scheduler process that controls the interaction's
    functions: (getting observations, emitting action, updating the agent)
    """
    _logger = logging.getLogger(__name__)

    socket = context.socket(zmq.PUB)
    socket.bind(event_bus_cfg.scheduler_connection)
    topic = EventTopic.corerl_scheduler
    message_data = Event(type=EventType.step).model_dump_json()

    while not stop_event.is_set():
        payload = f"{topic} {message_data}"
        _logger.debug(payload)
        socket.send_string(payload)
        time.sleep(1)
