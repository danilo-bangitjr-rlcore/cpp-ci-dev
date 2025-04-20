import logging
import threading
import time
from datetime import UTC, datetime, timedelta

import zmq

from corerl.messages.event_bus import DummyEventBus, EventBus
from corerl.messages.events import Event, EventTopic, EventType
from corerl.state import AppState

logger = logging.getLogger(__name__)


def start_scheduler_thread(app_state: AppState):
    scheduler_thread = threading.Thread(
        target=scheduler_task,
        args=(app_state, ),
        daemon=True,
        name= "corerl_interaction_scheduler",
    )
    scheduler_thread.start()
    return scheduler_thread


def scheduler_task(app_state: AppState):
    """
    Thread worker that emits ZMQ messages using our messages Event class.
    Responsible for emitting the step events based on configured observation windows.
    """
    cfg = app_state.cfg.env
    action_clock = Clock(EventType.step_emit_action, cfg.action_period, offset=timedelta(seconds=1))
    clocks = [
        action_clock,
        Clock(EventType.step_get_obs, cfg.obs_period),
        Clock(EventType.step_agent_update, cfg.update_period),
        Clock(EventType.agent_step, cfg.obs_period),
    ]

    if cfg.setpoint_ping_period is not None:
        clocks += [
            Clock(EventType.ping_setpoints, cfg.setpoint_ping_period, offset=cfg.setpoint_ping_period),
        ]

    app_state.event_bus.attach_callback(
        EventType.action_period_reset,
        lambda _: action_clock.reset(datetime.now(UTC)),
    )

    while not app_state.stop_event.is_set():
        try:
            now = datetime.now(UTC)
            for clock in clocks:
                clock.maybe_emit(app_state.event_bus, now)

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

    def emit(self, event_bus: EventBus | DummyEventBus, now: datetime):
        event = Event(type=self._event_type)
        try:
            event_bus.emit_event(event, topic=EventTopic.corerl_scheduler)
        except zmq.ZMQError as e:
            if isinstance(e, zmq.error.Again):
                # temporarily unavailable, retry
                return
            else:
                raise

        self.reset(now)

    def should_emit(self, now: datetime):
        return now > self._next_ts

    def maybe_emit(self, event_bus: EventBus | DummyEventBus, now: datetime):
        if self.should_emit(now):
            self.emit(event_bus, now)

    def get_next_ts(self):
        return self._next_ts

    def reset(self, now: datetime):
        self._next_ts = now + self._period
