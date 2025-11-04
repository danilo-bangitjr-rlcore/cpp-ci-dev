import socket
import time
from enum import auto

import pytest
from lib_defs.type_defs.base_events import BaseEvent, BaseEventTopic, BaseEventType
from lib_utils.time import now_iso
from pydantic import Field

from lib_events.client.event_bus_client import EventBusClient
from lib_events.server.proxy import EventBusProxy


def get_free_port(host: str = "localhost") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        s.listen(1)
        return s.getsockname()[1]


class TestEventType(BaseEventType):
    test_start = auto()
    test_update = auto()
    test_stop = auto()


class TestEventTopic(BaseEventTopic):
    test_topic = auto()
    debug_topic = auto()


class TestEvent(BaseEvent[TestEventType]):
    time: str = Field(default_factory=now_iso)
    type: TestEventType


@pytest.fixture
def broker_port() -> int:
    """
    Generate unique broker port for test to avoid conflicts.
    """
    return get_free_port("localhost")


@pytest.fixture
def broker(broker_port: int):
    """
    Create real EventBusProxy broker for testing.
    """
    proxy = EventBusProxy(router_addr=f"tcp://127.0.0.1:{broker_port}")
    proxy.start()
    time.sleep(0.2)

    yield proxy

    if proxy.is_running():
        proxy.stop()


@pytest.fixture
def client(broker: EventBusProxy, broker_port: int):
    """
    Create EventBusClient connected to broker.
    """
    client = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
        max_reconnect_attempts=3,
    )
    yield client
    try:
        if client.is_connected():
            client.close()
    except Exception:
        pass


def test_client_initial_state(broker_port: int):
    """
    Client starts in disconnected state.
    """
    client = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    assert not client.is_connected()


@pytest.mark.timeout(10)
def test_client_connect_and_close(client: EventBusClient):
    """
    Client can connect and close cleanly.
    """
    client.connect()
    time.sleep(0.1)
    assert client.is_connected()

    client.close()

    assert not client.is_connected()


def test_client_emit_without_connect(broker_port: int):
    """
    Emitting without connection logs warning and does nothing.
    """
    client = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    assert not client.is_connected()

    event = TestEvent(type=TestEventType.test_start)
    client.emit_event(event, topic=TestEventTopic.test_topic)


@pytest.mark.timeout(10)
def test_client_idempotent_connect(client: EventBusClient):
    """
    Multiple connect calls are safe.
    """
    client.connect()
    time.sleep(0.1)
    assert client.is_connected()

    client.connect()

    assert client.is_connected()



@pytest.mark.timeout(10)
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



@pytest.mark.timeout(10)
def test_client_pub_sub_message_flow(broker: EventBusProxy, broker_port: int):
    """
    Messages published by one client are received by subscribing client.
    """
    publisher = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    subscriber.connect()
    subscriber.subscribe(TestEventTopic.test_topic)
    subscriber.start_consumer()
    time.sleep(0.2)

    event = TestEvent(type=TestEventType.test_start)
    publisher.emit_event(event, topic=TestEventTopic.test_topic)
    time.sleep(0.5)

    received_event = subscriber.recv_event()
    assert received_event is not None
    assert received_event.type == TestEventType.test_start

    publisher.close()
    subscriber.close()



@pytest.mark.timeout(10)
def test_client_callback_invocation(broker: EventBusProxy, broker_port: int):
    """
    Attached callbacks are invoked when matching events are received.
    """
    publisher = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    subscriber.connect()
    time.sleep(0.1)

    callback_invoked = []

    def test_callback(event: TestEvent):
        callback_invoked.append(event)

    subscriber.subscribe(TestEventTopic.test_topic)
    subscriber.attach_callback(TestEventType.test_start, test_callback)
    subscriber.start_consumer()
    time.sleep(0.2)

    event = TestEvent(type=TestEventType.test_start)
    publisher.emit_event(event, topic=TestEventTopic.test_topic)
    time.sleep(0.5)

    assert len(callback_invoked) == 1
    assert callback_invoked[0].type == TestEventType.test_start

    publisher.close()
    subscriber.close()



@pytest.mark.timeout(10)
def test_client_multiple_callbacks(broker: EventBusProxy, broker_port: int):
    """
    Multiple callbacks can be attached to the same event type.
    """
    publisher = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    subscriber.connect()
    time.sleep(0.1)

    callback1_invoked = []
    callback2_invoked = []

    def callback1(event: TestEvent):
        callback1_invoked.append(event)

    def callback2(event: TestEvent):
        callback2_invoked.append(event)

    subscriber.subscribe(TestEventTopic.test_topic)
    subscriber.attach_callback(TestEventType.test_start, callback1)
    subscriber.attach_callback(TestEventType.test_start, callback2)
    subscriber.start_consumer()
    time.sleep(0.2)

    event = TestEvent(type=TestEventType.test_start)
    publisher.emit_event(event, topic=TestEventTopic.test_topic)
    time.sleep(0.5)

    assert len(callback1_invoked) == 1
    assert len(callback2_invoked) == 1

    publisher.close()
    subscriber.close()



@pytest.mark.timeout(10)
def test_client_topic_filtering(broker: EventBusProxy, broker_port: int):
    """
    Subscribers only receive messages from topics they subscribe to.
    """
    publisher = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    subscriber.connect()
    subscriber.subscribe(TestEventTopic.test_topic)
    subscriber.start_consumer()
    time.sleep(0.2)

    event_wrong_topic = TestEvent(type=TestEventType.test_start)
    publisher.emit_event(event_wrong_topic, topic=TestEventTopic.debug_topic)
    time.sleep(0.3)

    received = subscriber.recv_event(timeout=0.5)
    assert received is None

    event_correct_topic = TestEvent(type=TestEventType.test_update)
    publisher.emit_event(event_correct_topic, topic=TestEventTopic.test_topic)
    time.sleep(0.3)

    received = subscriber.recv_event(timeout=0.5)
    assert received is not None
    assert received.type == TestEventType.test_update

    publisher.close()
    subscriber.close()



@pytest.mark.timeout(10)
def test_client_emit_event_type_shortcut(broker: EventBusProxy, broker_port: int):
    """
    Can emit just an event type, which gets wrapped in event automatically.
    """
    publisher = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    subscriber.connect()
    subscriber.subscribe(TestEventTopic.test_topic)
    subscriber.start_consumer()
    time.sleep(0.2)

    publisher.emit_event(TestEventType.test_start, topic=TestEventTopic.test_topic)
    time.sleep(0.5)

    received_event = subscriber.recv_event()
    assert received_event is not None
    assert received_event.type == TestEventType.test_start

    publisher.close()
    subscriber.close()



@pytest.mark.timeout(10)
def test_client_reconnection_preserves_state(
    broker: EventBusProxy,

    broker_port: int,
):
    """
    Subscriptions and state are preserved when client reconnects manually.
    """
    client = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    client.connect()
    client.subscribe(TestEventTopic.test_topic)
    client.subscribe(TestEventTopic.debug_topic)

    assert len(client._subscribed_topics) == 2
    assert TestEventTopic.test_topic in client._subscribed_topics
    assert TestEventTopic.debug_topic in client._subscribed_topics

    original_topics = client._subscribed_topics.copy()

    client.close()
    time.sleep(0.1)
    client.connect()

    assert len(client._subscribed_topics) == 2
    assert client._subscribed_topics == original_topics

    client.close()


def test_client_max_reconnect_attempts():
    """
    Client respects max reconnect attempts configuration by stopping after limit reached.

    Note: ZMQ connect() succeeds even without a listening server, so we manually
    set _connected=False and track that the counter properly caps at max_reconnect_attempts.
    """
    free_port = get_free_port("localhost")

    client = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=free_port,
        max_reconnect_attempts=2,
        reconnect_interval=0.1,
    )

    client.connect()
    assert client._reconnect_attempts == 0

    client._connected = False
    client._reconnect_attempts = 0
    result = client._reconnect()
    assert result is True

    client._connected = False
    client._reconnect_attempts = 1
    result = client._reconnect()
    assert result is True

    client._connected = False
    client._reconnect_attempts = 2
    result = client._reconnect()
    assert result is False
    assert client._reconnect_attempts == 2

    client.stop_event.set()
    client.close()



@pytest.mark.timeout(10)
def test_client_reconnect_preserves_subscriptions(
    broker: EventBusProxy,

    broker_port: int,
):
    """
    Subscriptions are re-established after reconnection.
    """
    client = EventBusClient[TestEvent, TestEventType, TestEventTopic](
        event_class=TestEvent,
        host="127.0.0.1",
        port=broker_port,
    )
    client.connect()
    client.subscribe(TestEventTopic.test_topic)
    client.subscribe(TestEventTopic.debug_topic)

    assert len(client._subscribed_topics) == 2
    assert TestEventTopic.test_topic in client._subscribed_topics
    assert TestEventTopic.debug_topic in client._subscribed_topics

    client.close()
    client.connect()

    assert len(client._subscribed_topics) == 2

    client.close()
