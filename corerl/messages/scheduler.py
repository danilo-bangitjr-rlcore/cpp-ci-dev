import logging
import threading
import time
from collections import namedtuple
from datetime import UTC, datetime

import zmq

from corerl.config import MainConfig
from corerl.messages.events import Event, EventTopic, EventType


def scheduler_task(cfg: MainConfig, context: zmq.Context, stop_event: threading.Event):
    """
    Thread worker that emits ZMQ messages using our messages Event class.
    Responsible for emitting the step events based on configured observation windows.
    """
    _logger = logging.getLogger(__name__)

    socket = context.socket(zmq.PUB)
    socket.bind(cfg.event_bus.scheduler_connection)
    topic = EventTopic.corerl_scheduler

    EventPeriodTime = namedtuple('EventPeriodTime', ['period', 'next_event_ts'])

    thread_start_time = datetime.now(UTC)
    next_event_dict = {
        EventType.step_get_obs: EventPeriodTime(cfg.env.obs_period, thread_start_time + cfg.env.obs_period),
        EventType.step_agent_update: EventPeriodTime(cfg.env.update_period, thread_start_time + cfg.env.update_period),
        EventType.step_emit_action: EventPeriodTime(cfg.env.action_period, thread_start_time + cfg.env.action_period),
    }

    def _write_to_zmq(event: Event):
        message_data = event.model_dump_json()
        payload = f"{topic} {message_data}"
        _logger.debug(payload)
        socket.send_string(payload)

    while not stop_event.is_set():
        thread_cur_time = datetime.now(UTC)

        for event_type, ept in next_event_dict.items():
            if thread_cur_time >= ept.next_event_ts:
                _write_to_zmq(Event(type=event_type))
                next_event_dict[event_type] = EventPeriodTime(
                    period = ept.period,
                    next_event_ts= ept.next_event_ts + ept.period
                )

        time.sleep(0.5)
