import logging
import threading
import time
from datetime import UTC, datetime, timedelta

import zmq

from corerl.environment.async_env.factory import AsyncEnvConfig
from corerl.messages.events import Event, EventTopic, EventType

logger = logging.getLogger(__name__)


def scheduler_task(pub_socket: zmq.Socket, cfg: AsyncEnvConfig, stop_event: threading.Event):
    """
    Thread worker that emits ZMQ messages using our messages Event class.
    Responsible for emitting the step events based on configured observation windows.
    """

    clocks = [
        Clock(EventType.step_get_obs, cfg.obs_period),
        Clock(EventType.step_agent_update, cfg.update_period),
        Clock(EventType.step_emit_action, cfg.action_period, offset=timedelta(seconds=1)),
        Clock(EventType.agent_step, cfg.obs_period),
    ]

    if cfg.setpoint_ping_period is not None:
        clocks += [
            Clock(EventType.ping_setpoints, cfg.setpoint_ping_period, offset=cfg.setpoint_ping_period),
        ]

    while not stop_event.is_set():
        try:
            now = datetime.now(UTC)
            for clock in clocks:
                clock.maybe_emit(pub_socket, now)

            shortest_duration = min(clock.get_next_ts() for clock in clocks) - datetime.now(UTC)
            shortest_duration = max(shortest_duration.total_seconds(), 0)
            time.sleep(shortest_duration)

        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                # exit, break from loop
                break
            raise


class Clock:
    def __init__(self, event_type: EventType, period: timedelta, offset: timedelta = timedelta(seconds=0)):
        self._event_type = event_type
        self._period = period

        self._next_ts = datetime.now(UTC) + offset

    def emit(self, socket: zmq.Socket, now: datetime):
        event = Event(type=self._event_type)
        logger.debug(f"Scheduling Event: {event}")
        message_data = event.model_dump_json()
        payload = f"{EventTopic.corerl_scheduler} {message_data}"

        try:
            socket.send_string(payload)
        except zmq.ZMQError as e:
            if isinstance(e, zmq.error.Again):
                # temporarily unavailable, retry
                return
            else:
                raise

        self.reset(now)

    def should_emit(self, now: datetime):
        return now > self._next_ts

    def maybe_emit(self, socket: zmq.Socket, now: datetime):
        if self.should_emit(now):
            self.emit(socket, now)

    def get_next_ts(self):
        return self._next_ts

    def reset(self, now: datetime):
        self._next_ts = now + self._period
