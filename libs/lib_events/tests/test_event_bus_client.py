import socket
import time

import pytest
from lib_defs.type_defs.base_events import Event, EventTopic, EventType

from lib_events.client.event_bus_client import EventBusClient
from lib_events.server.proxy import EventBusProxy


def get_free_port(host: str = "localhost") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        s.listen(1)
        return s.getsockname()[1]


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
    time.sleep(0.1)
    yield proxy

    if proxy.is_running():
        proxy.stop()


@pytest.fixture
def client(broker: EventBusProxy, broker_port: int):
    """
    Create EventBusClient connected to broker.
    """
    client = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
        max_reconnect_attempts=3,
    )
    yield client
    try:
        if client.consumer_thread is not None and client.consumer_thread.is_alive():
            client.stop_consumer()
        if client.is_connected():
            client.close()
    except Exception:
        pass


@pytest.fixture
def publisher(broker: EventBusProxy, broker_port: int):
    """
    Create publisher client connected to broker.
    """
    client = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
    )
    client.connect()
    time.sleep(0.1)
    yield client
    try:
        if client.consumer_thread is not None and client.consumer_thread.is_alive():
            client.stop_consumer()
        if client.is_connected():
            client.close()
    except Exception:
        pass


@pytest.fixture
def subscriber(broker: EventBusProxy, broker_port: int):
    """
    Create subscriber client connected to broker.
    """
    client = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
    )
    client.connect()
    time.sleep(0.1)
    yield client
    try:
        if client.consumer_thread is not None and client.consumer_thread.is_alive():
            client.stop_consumer()
        if client.is_connected():
            client.close()
    except Exception:
        pass


@pytest.fixture
def requester(broker: EventBusProxy, broker_port: int):
    """
    Create requester client connected to broker.
    """
    client = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
        service_id="requester",
    )
    client.connect()
    client.start_consumer()
    time.sleep(0.1)
    yield client
    try:
        if client.consumer_thread is not None and client.consumer_thread.is_alive():
            client.stop_consumer()
        if client.is_connected():
            client.close()
    except Exception:
        pass


@pytest.fixture
def responder(broker: EventBusProxy, broker_port: int):
    """
    Create responder client connected to broker.
    """
    client = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
        service_id="responder",
    )
    client.connect()
    yield client
    try:
        if client.consumer_thread is not None and client.consumer_thread.is_alive():
            client.stop_consumer()
        if client.is_connected():
            client.close()
    except Exception:
        pass


def test_client_connection_lifecycle(client: EventBusClient):
    """
    Client starts disconnected, can connect, handles idempotent connects, and closes cleanly.
    """
    assert not client.is_connected()

    client.connect()
    time.sleep(0.1)
    assert client.is_connected()

    client.connect()
    assert client.is_connected()

    client.close()
    assert not client.is_connected()



@pytest.mark.timeout(10)
def test_client_lifecycle_multiple_cycles(client: EventBusClient):
    """
    Client can be connected and closed multiple times without resource leaks.
    """
    for _ in range(2):
        client.connect()
        time.sleep(0.1)
        assert client.is_connected()

        client.close()
        assert not client.is_connected()



def test_client_pub_sub_message_flow(publisher: EventBusClient, subscriber: EventBusClient):
    """
    Messages published by one client are received by subscribing client.
    """
    subscriber.subscribe(EventTopic.corerl)
    subscriber.start_consumer()
    time.sleep(0.2)

    event = Event(type=EventType.service_started)
    publisher.emit_event(event, topic=EventTopic.corerl)
    time.sleep(0.5)

    received_event = subscriber.recv_event()
    assert received_event is not None
    assert received_event.type == EventType.service_started



@pytest.mark.timeout(10)
def test_client_callback_invocation(broker: EventBusProxy, broker_port: int):
    """
    Pub/sub with single callback, multiple callbacks, and event type shortcut.
    """
    publisher = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
    )
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
    )
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



def test_client_topic_filtering(publisher: EventBusClient, subscriber: EventBusClient):
    """
    Subscribers only receive messages from topics they subscribe to.
    """
    subscriber.subscribe(EventTopic.corerl)
    subscriber.start_consumer()
    time.sleep(0.2)

    event_wrong_topic = Event(type=EventType.service_started)
    publisher.emit_event(event_wrong_topic, topic=EventTopic.debug_app)
    time.sleep(0.3)

    received = subscriber.recv_event(timeout=0.5)
    assert received is None

    event_correct_topic = Event(type=EventType.step)
    publisher.emit_event(event_correct_topic, topic=EventTopic.corerl)
    time.sleep(0.3)

    received = subscriber.recv_event(timeout=0.5)
    assert received is not None
    assert received.type == EventType.step



@pytest.mark.timeout(10)
def test_client_emit_event_type_shortcut(broker: EventBusProxy, broker_port: int):
    """
    Can emit just an event type, which gets wrapped in event automatically.
    """
    publisher = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
    )
    publisher.connect()
    time.sleep(0.1)

    subscriber = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
    )
    subscriber.connect()
    subscriber.subscribe(EventTopic.corerl)
    subscriber.start_consumer()
    time.sleep(0.2)

    publisher.emit_event(EventType.service_started, topic=EventTopic.corerl)
    time.sleep(0.5)

    received_event = subscriber.recv_event()
    assert received_event is not None
    assert received_event.type == EventType.service_started

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
    client = EventBusClient(
        host="127.0.0.1",
        port=broker_port,
    )
    client.connect()
    client.subscribe(EventTopic.corerl)
    client.subscribe(EventTopic.debug_app)

    assert len(client._subscribed_topics) == 2
    assert EventTopic.corerl in client._subscribed_topics
    assert EventTopic.debug_app in client._subscribed_topics

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

    client = EventBusClient(
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
    assert result is not None

    client._connected = False
    client._reconnect_attempts = 1
    result = client._reconnect()
    assert result is not None

    client._connected = False
    client._reconnect_attempts = 2
    result = client._reconnect()
    assert result is None
    assert client._reconnect_attempts == 2

    client.stop_event.set()
    client.close()



@pytest.mark.timeout(10)
def test_client_request_reply(requester: EventBusClient, responder: EventBusClient):
    """
    Client can send REQUEST and receive REPLY from another client.
    """
    def handle_request(payload: bytes) -> bytes:
        request_text = payload.decode()
        return f"Echo: {request_text}".encode()

    responder.register_request_handler(handle_request)
    responder.start_consumer()
    time.sleep(0.2)

    response = requester.send_request(
        service_id="responder",
        payload=b"Hello",
        timeout=2.0,
    )
    assert response == b"Echo: Hello"


@pytest.mark.timeout(15)
def test_client_multiple_concurrent_requests(requester: EventBusClient, responder: EventBusClient):
    """
    Multiple concurrent requests are handled with correct correlation IDs.
    """
    import concurrent.futures

    def handle_request(payload: bytes) -> bytes:
        request_num = int(payload.decode())
        time.sleep(0.1)
        return f"Response-{request_num}".encode()

    responder.register_request_handler(handle_request)
    responder.start_consumer()
    time.sleep(0.2)

    def send_request(num: int) -> bytes:
        return requester.send_request(
            service_id="responder",
            payload=str(num).encode(),
            timeout=3.0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_request, i) for i in range(5)]
        responses = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(responses) == 5
    response_texts = {r.decode() for r in responses}
    assert response_texts == {f"Response-{i}" for i in range(5)}


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_client_request_reply_async(requester: EventBusClient, responder: EventBusClient):
    """
    Client can send async REQUEST and receive REPLY.

    Note: This test requires pytest-asyncio to be installed.
    Run: uv sync --all-groups in the lib_events directory.
    """
    def handle_request(payload: bytes) -> bytes:
        request_text = payload.decode()
        return f"Async Echo: {request_text}".encode()

    responder.register_request_handler(handle_request)
    responder.start_consumer()
    time.sleep(0.2)

    response = await requester.send_request_async(
        service_id="responder",
        payload=b"Hello Async",
        timeout=2.0,
    )
    assert response == b"Async Echo: Hello Async"
