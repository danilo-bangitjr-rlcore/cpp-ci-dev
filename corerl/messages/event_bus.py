import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import zmq

from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.messages.consumer import consumer_task
from corerl.messages.events import Event, EventTopic, EventType
from corerl.messages.factory import EventBusConfig
from corerl.messages.scheduler import scheduler_task

_logger = logging.getLogger(__name__)

@dataclass
class EventBusState:
    scheduler_thread: None | threading.Thread = None
    subscriber_socket: None | zmq.Socket = None
    publisher_socket: None | zmq.Socket = None



Callback = Callable[[Event], Any]


class EventBus:
    """EventBus enables asynchronous communication through a ZMQ pub-sub messaging pattern.
    Spins up the scheduler thread, the consumer thread, and the FIFO subscriber queue.
    """
    def __init__(self, cfg_event_bus: EventBusConfig, cfg_env: AsyncEnvConfig):
        self.cfg_event_bus = cfg_event_bus
        self.cfg_env = cfg_env

        self.queue = Queue()
        self.zmq_context = zmq.Context()
        self.subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.publisher_socket = self.zmq_context.socket(zmq.PUB)

        self.event_bus_stop_event = threading.Event()
        self.consumer_thread = threading.Thread(
            target=consumer_task,
            args=(
                self.subscriber_socket,
                self.queue,
                self.event_bus_stop_event
            ),
            daemon=True,
            name= "corerl_event_bus_consumer",
        )
        self.scheduler_thread = threading.Thread(
            target=scheduler_task,
            args=(
                self.publisher_socket,
                self.cfg_env,
                self.event_bus_stop_event
            ),
            daemon=True,
            name= "corerl_event_bus_scheduler",
        )

        self.subscriber_socket.bind(self.cfg_event_bus.app_connection)
        self.subscriber_socket.bind(self.cfg_event_bus.cli_connection)
        self.publisher_socket.connect(self.cfg_event_bus.app_connection)

        self._callbacks: dict[EventType, list[Callback]] = defaultdict(list)


    def start(self):
        self.consumer_thread.start()
        self.scheduler_thread.start()

    def emit_event(self, event: Event | EventType, topic: EventTopic = EventTopic.debug_app):
        if isinstance(event, EventType):
            event = Event(type=event)

        message_data = event.model_dump_json()
        self.publisher_socket.send_string(f"{topic} {message_data}")

    def recv_event(self) -> None | Event:
        if self.event_bus_stop_event.is_set():
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


    def listen_forever(self, max_steps: int | None = None):
        steps = 0
        while True:
            event = self.recv_event()
            if event is None:
                continue

            for cb in self._callbacks[event.type]:
                cb(event)

            yield event
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break


    def attach_callback(self, event_type: EventType, cb: Callback):
        self._callbacks[event_type].append(cb)


    def attach_callbacks(self, cbs: dict[EventType, Callback]):
        for event_type, cb in cbs.items():
            self.attach_callback(event_type, cb)


    def cleanup(self):
        self.event_bus_stop_event.set()
        self.scheduler_thread.join()

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
        self.publisher_socket.close()
        self.zmq_context.term()

        _logger.info("Cleaned up event bus")


class DummyEventBus:
    def listen_forever(self, max_steps: int | None = None):
        steps = 0
        while True:
            yield
            steps += 1

            if max_steps is not None and steps >= max_steps:
                break

    def start(self): ...
    def attach_callback(self, event_type: EventType, cb: Callback): ...
    def attach_callbacks(self, cbs: dict[EventType, Callback]): ...
    def emit_event(self, event: Event | EventType, topic: EventTopic = EventTopic.debug_app): ...
    def cleanup(self): ...
