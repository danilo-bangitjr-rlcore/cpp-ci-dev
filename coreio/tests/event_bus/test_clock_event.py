import threading
from datetime import UTC, datetime, timedelta
from math import isclose

from lib_utils.messages.base_event_bus import BaseEventBus

from coreio.communication.scheduler import start_scheduler_io_thread
from coreio.config import DataIngressConfig
from coreio.utils.io_events import IOEvent, IOEventTopic, IOEventType


def test_receive_event():
    """
    Test that scheduler sends IOEvent.read_opcua_nodes at correct intervals,
    and that the timing accuracy is within 1% tolerance.
    """
    coreio_app_addr = "inproc://test_coreio_app"
    zmq_communication = BaseEventBus(
        event_class = IOEvent,
        topic = IOEventTopic.coreio,
        consumer_name = "test_coreio_consumer",
        subscriber_addrs=[coreio_app_addr],
        publisher_addr=coreio_app_addr,
    )
    zmq_communication.start()
    ingress_stop_event = threading.Event()

    # Start scheduler with read event sent every 200 ms
    ingress_period = timedelta(seconds=0.2)
    data_ingress_cfg =  DataIngressConfig(enabled=True, ingress_period=ingress_period)
    start_scheduler_io_thread(data_ingress_cfg, ingress_stop_event, zmq_communication)

    start_time = datetime.now(UTC)
    for i in range(10):
        event = zmq_communication.recv_event()
        if event is None:
            continue

        assert event.type == IOEventType.read_opcua_nodes
        end_time = datetime.now(UTC)
        measured_period = end_time - start_time
        start_time = datetime.now(UTC)

        # Skip first reading, then check the period of received messages
        if i > 0:
            assert isclose(measured_period.total_seconds(), ingress_period.total_seconds(), rel_tol=0.01)

    ingress_stop_event.set()
    zmq_communication.cleanup()

