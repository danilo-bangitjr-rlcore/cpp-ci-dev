import threading
import time

import pytest
import zmq
from lib_defs.type_defs.base_events import Event, EventTopic, EventType
from test.infrastructure.networking import get_free_port

from corerl.event_bus.client import EventBusClient


@pytest.fixture
def pub_port() -> int:
    """
    Generate unique publisher port for test to avoid conflicts.
    """
    return get_free_port("localhost")


@pytest.fixture
def sub_port() -> int:
    """
    Generate unique subscriber port for test to avoid conflicts.
    """
    return get_free_port("localhost")


@pytest.fixture
def mock_proxy(pub_port: int, sub_port: int):
    """
    Create mock proxy with XSUB socket for publisher and XPUB for subscriber that forwards messages.
    """
    context = zmq.Context()

    xsub_socket = context.socket(zmq.XSUB)
    xsub_socket.bind(f"tcp://127.0.0.1:{pub_port}")

    xpub_socket = context.socket(zmq.XPUB)
    xpub_socket.bind(f"tcp://127.0.0.1:{sub_port}")

    stop_event = threading.Event()

    def forward_messages():
        poller = zmq.Poller()
        poller.register(xsub_socket, zmq.POLLIN)
        poller.register(xpub_socket, zmq.POLLIN)

        while not stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=100))
            except zmq.ZMQError:
                if stop_event.is_set():
                    break
                continue

            if xsub_socket in socks:
                message = xsub_socket.recv_multipart(zmq.NOBLOCK)
                xpub_socket.send_multipart(message)

            if xpub_socket in socks:
                message = xpub_socket.recv_multipart(zmq.NOBLOCK)
                xsub_socket.send_multipart(message)

    proxy_thread = threading.Thread(target=forward_messages, daemon=True)
    proxy_thread.start()

    yield xsub_socket, xpub_socket

    stop_event.set()
    proxy_thread.join(timeout=2)
    xsub_socket.close()
    xpub_socket.close()
    context.term()


@pytest.fixture
def client(mock_proxy: tuple[zmq.Socket, zmq.Socket], pub_port: int, sub_port: int):
    """
    Create EventBusClient connected to mock proxy.
    """
    client = EventBusClient(host="127.0.0.1", pub_port=pub_port, sub_port=sub_port)
    yield client
    if client.is_connected():
        client.close()


@pytest.mark.timeout(5)
def test_client_initial_state(pub_port: int, sub_port: int):
    """
    Client starts in disconnected state.
    """
    client = EventBusClient(host="127.0.0.1", pub_port=pub_port, sub_port=sub_port)
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
def test_client_emit_without_connect(pub_port: int, sub_port: int):
    """
    Emitting without connection logs warning and does nothing.
    """
    client = EventBusClient(host="127.0.0.1", pub_port=pub_port, sub_port=sub_port)
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
def test_client_callback_invocation(mock_proxy: tuple[zmq.Socket, zmq.Socket], pub_port: int, sub_port: int):
    """
    Attached callbacks are invoked when matching events are received.
    """
    publisher = EventBusClient(host="127.0.0.1", pub_port=pub_port, sub_port=sub_port)
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient(host="127.0.0.1", pub_port=pub_port, sub_port=sub_port)
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
