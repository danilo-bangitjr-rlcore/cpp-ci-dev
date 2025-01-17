import time

import zmq

from corerl.messages.events import Event, EventType
from corerl.messages.factory import MessageBusConfig


def scheduler_task(message_bus_cfg: MessageBusConfig):
    """
    Processs that emits ZMQ messages using our messages Event class.
    Stub for the CoreRL Scheduler process that controls the interaction's
    functions: (getting observations, emitting action, updating the agent)
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(message_bus_cfg.scheduler_connection)
    counter = 0
    topics = ["system", "test"]
    message_data = Event(type=EventType.heartbeat).model_dump_json()

    while True:
        topic = topics[counter % 2]
        payload = f"{topic} {message_data}"
        print(payload)
        socket.send_string(payload)
        counter += 1
        time.sleep(1)
