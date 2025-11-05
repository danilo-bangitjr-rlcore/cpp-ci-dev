import time

import pytest
from lib_defs.type_defs.base_events import Event, EventTopic, EventType
from lib_events.server.proxy import EventBusProxy
from test.infrastructure.networking import get_free_port

from corerl.event_bus.client import EventBusClient


@pytest.fixture
def comms_port() -> int:
    """
    Generate unique communications port for test to avoid conflicts.
    """
    return get_free_port("localhost")


@pytest.fixture
def event_bus_proxy(comms_port: int):
    """
    Create real EventBusProxy for testing.
    """
    proxy = EventBusProxy(router_addr=f"tcp://127.0.0.1:{comms_port}")
    proxy.start()
    time.sleep(0.1)
    yield proxy
    proxy.stop()


@pytest.fixture
def client(event_bus_proxy: EventBusProxy, comms_port: int):
    """
    Create EventBusClient connected to real proxy using dynamic port.
    """
    client = EventBusClient(host="127.0.0.1", port=comms_port)
    yield client
    if client.is_connected():
        client.close()


@pytest.mark.timeout(5)
def test_client_initial_state(comms_port: int):
    """
    Client starts in disconnected state.
    """
    client = EventBusClient(host="127.0.0.1", port=comms_port)
    assert not client.is_connected()


@pytest.mark.timeout(5)
def test_client_close(client: EventBusClient):
    """
    Client can be closed cleanly after connecting.
    """
    client.connect()
    time.sleep(0.1)
    assert client.is_connected()

    client.close()

    assert not client.is_connected()


@pytest.mark.timeout(5)
def test_client_emit_without_connect(comms_port: int):
    """
    Emitting without connection logs warning and does nothing.
    """
    client = EventBusClient(host="127.0.0.1", port=comms_port)
    assert not client.is_connected()

    event = Event(type=EventType.service_started)
    client.emit_event(event, topic=EventTopic.corerl)


@pytest.mark.timeout(5)
def test_client_idempotent_connect(client: EventBusClient):
    """
    Multiple connect calls are safe.
    """
    client.connect()
    time.sleep(0.1)
    assert client.is_connected()

    client.connect()

    assert client.is_connected()


@pytest.mark.timeout(5)
def test_client_lifecycle_multiple_cycles(client: EventBusClient):
    """
    Client can be connected and closed multiple times.
    """
    for _ in range(3):
        client.connect()
        time.sleep(0.1)
        assert client.is_connected()

        client.close()
        assert not client.is_connected()


@pytest.mark.timeout(5)
def test_client_callback_invocation(event_bus_proxy: EventBusProxy, comms_port: int):
    """
    Attached callbacks are invoked when matching events are received.

    Note: This test validates the unified event bus client wrapper but does not reflect
    the actual CoreRL deployment which uses legacy EventBus for peer-to-peer communication.
    """
    publisher = EventBusClient(host="127.0.0.1", port=comms_port)
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient(host="127.0.0.1", port=comms_port)
    subscriber.connect()
    time.sleep(0.1)

    callback_invoked = []

    def test_callback(event: Event):
        callback_invoked.append(event)

    subscriber.subscribe(EventTopic.corerl)
    subscriber.attach_callback(EventType.service_started, test_callback)
    subscriber.start_consumer()
    time.sleep(0.2)

    event = Event(type=EventType.service_started)
    publisher.emit_event(event, topic=EventTopic.corerl)
    time.sleep(0.5)

    assert len(callback_invoked) == 1
    assert callback_invoked[0].type == EventType.service_started

    publisher.close()
    subscriber.close()
