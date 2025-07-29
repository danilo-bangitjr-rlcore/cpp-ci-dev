import threading
import time
from datetime import UTC, datetime, timedelta

import zmq
from lib_utils.messages.base_event_bus import BaseEventBus
from lib_utils.messages.clock import Clock

from coreio.config import DataIngressConfig
from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType


def start_scheduler_io_thread(cfg: DataIngressConfig, stop_event: threading.Event, EventBus: BaseEventBus):
    scheduler_io_thread = threading.Thread(
        target=scheduler_io_task,
        args=(cfg, stop_event, EventBus),
        daemon=True,
        name="coreio_scheduler",
    )
    scheduler_io_thread.start()
    return scheduler_io_thread

def create_scheduler_io_clock(
        event_type: IOEventType,
        period: timedelta,
        offset: timedelta = timedelta(seconds=0),
) -> Clock[IOEvent, IOEventTopic, IOEventType]:
    """
    Factory function to create clocks specifically for the CoreIO scheduler topic
    """
    return  Clock(IOEvent, IOEventTopic.coreio, event_type, period, offset)

def scheduler_io_task(cfg: DataIngressConfig, stop_event: threading.Event, EventBus: BaseEventBus):
    data_ingress_clock = create_scheduler_io_clock(IOEventType.read_opcua_nodes, cfg.ingress_period)

    while not stop_event.is_set():
        try:
            now = datetime.now(UTC)
            data_ingress_clock.maybe_emit(EventBus, now)
            time_sleep = (data_ingress_clock.get_next_ts() - datetime.now(UTC)).total_seconds()
            time_sleep = max(time_sleep, 0)
            time.sleep(time_sleep) # Maybe async
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                # exit, break from loop
                break
            raise

