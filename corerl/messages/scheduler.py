import time
import zmq
from corerl.messages.factory import MessageBusConfig
from corerl.messages.events import Event, EventType


def scheduler_task(message_bus_cfg: MessageBusConfig):
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


