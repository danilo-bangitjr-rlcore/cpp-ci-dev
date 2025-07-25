import logging
import threading
import time
from datetime import UTC, datetime, timedelta

import zmq
from lib_utils.messages.clock import Clock

from corerl.messages.event_bus import EventBus
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

def create_scheduler_clock(
        event_type: EventType,
        period: timedelta,
        offset: timedelta = timedelta(seconds=0),
) -> Clock[Event, EventTopic, EventType]:
    """
    Simple factory function, as all the Clocks here use Event and EventTopic.corerl_scheduler
    """
    return Clock(Event, EventTopic.corerl_scheduler, event_type, period, offset)

def scheduler_task(app_state: AppState[EventBus]):
    """
    Thread worker that emits ZMQ messages using our messages Event class.
    Responsible for emitting the step events based on configured observation windows.
    """
    cfg = app_state.cfg.interaction
    action_clock = create_scheduler_clock(EventType.step_emit_action, cfg.action_period, offset=timedelta(seconds=1))
    clocks = [
        action_clock,
        create_scheduler_clock(EventType.step_get_obs, cfg.obs_period),
        create_scheduler_clock(EventType.step_agent_update, cfg.update_period),
        create_scheduler_clock(EventType.agent_step, cfg.obs_period),
        create_scheduler_clock(EventType.flush_buffers, timedelta(seconds=30)),
    ]

    if cfg.setpoint_ping_period is not None:
        clocks += [
            create_scheduler_clock(EventType.ping_setpoints, cfg.setpoint_ping_period, offset=cfg.setpoint_ping_period),
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

