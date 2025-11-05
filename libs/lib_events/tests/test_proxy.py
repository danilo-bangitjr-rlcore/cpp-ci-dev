import socket
import time
from collections.abc import Callable

import pytest
import zmq

from lib_events.protocol.message_protocol import MessageType, build_message
from lib_events.server.proxy import EventBusProxy


@pytest.fixture
def unique_port():
    """
    Generate unique port number for each test to avoid conflicts.
    """
    def get_free_port():
        sock = socket.socket()
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    return get_free_port()


@pytest.fixture
def proxy(unique_port: int):
    """
    Create an EventBusProxy instance with unique port.
    """
    proxy = EventBusProxy(router_addr=f"tcp://127.0.0.1:{unique_port}")
    yield proxy
    if proxy.is_running():
        proxy.stop()


@pytest.fixture
def zmq_context():
    """
    Create ZMQ context for test clients.
    """
    ctx = zmq.Context()
    yield ctx
    ctx.term()


@pytest.fixture
def started_proxy(proxy: EventBusProxy):
    """
    Proxy already started and ready for connections.
    """
    proxy.start()
    time.sleep(0.1)
    return proxy


@pytest.fixture
def connected_client(zmq_context: zmq.Context, unique_port: int):
    """
    Factory for creating connected DEALER sockets with automatic cleanup.
    """
    sockets: list[zmq.Socket] = []

    def _create() -> zmq.Socket:
        sock = zmq_context.socket(zmq.DEALER)
        sock.connect(f"tcp://127.0.0.1:{unique_port}")
        sockets.append(sock)
        return sock

    yield _create

    for sock in sockets:
        sock.close()


def wait_for_message(socket: zmq.Socket, timeout_ms: int = 1000):
    """
    Poll socket for message. Returns (received, frames).
    """
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    socks = dict(poller.poll(timeout=timeout_ms))

    if socket in socks:
        return True, socket.recv_multipart()
    return False, None


def test_proxy_stop_terminates_thread(proxy: EventBusProxy):
    """
    Stopping proxy terminates thread and cleans up resources.
    """
    proxy.start()
    time.sleep(0.1)
    assert proxy.is_running()

    proxy.stop()

    assert not proxy.is_running()


def test_proxy_idempotent_stop(proxy: EventBusProxy):
    """
    Stopping an already stopped proxy does nothing.
    """
    proxy.stop()
    proxy.stop()
    assert not proxy.is_running()


def test_service_registration(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Service can register with broker using REGISTER message.
    """
    client = connected_client()

    register_msg = build_message(
        destination="test-service",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    client.send_multipart(register_msg)

    time.sleep(0.1)
    assert started_proxy.get_service_count() == 1


def test_topic_subscription(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Client can subscribe to topic using SUBSCRIBE message.
    """
    subscriber = connected_client()

    subscribe_msg = build_message(
        destination="test-topic",
        msg_type=MessageType.SUBSCRIBE,
        correlation_id="",
        payload=b"{}",
    )
    subscriber.send_multipart(subscribe_msg)

    time.sleep(0.1)
    assert started_proxy.get_topic_subscriber_count("test-topic") == 1


def test_publish_to_subscribers(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Published messages are delivered to all topic subscribers.
    """
    subscriber1 = connected_client()
    subscriber2 = connected_client()

    subscribe_msg = build_message(
        destination="events",
        msg_type=MessageType.SUBSCRIBE,
        correlation_id="",
        payload=b"{}",
    )
    subscriber1.send_multipart(subscribe_msg)
    subscriber2.send_multipart(subscribe_msg)

    time.sleep(0.1)

    publisher = connected_client()

    publish_msg = build_message(
        destination="events",
        msg_type=MessageType.PUBLISH,
        correlation_id="",
        payload=b'{"event": "test"}',
    )
    publisher.send_multipart(publish_msg)

    poller = zmq.Poller()
    poller.register(subscriber1, zmq.POLLIN)
    poller.register(subscriber2, zmq.POLLIN)

    received = 0
    deadline = time.time() + 2.0
    while received < 2 and time.time() < deadline:
        socks = dict(poller.poll(timeout=500))
        if subscriber1 in socks:
            subscriber1.recv_multipart()
            received += 1
        if subscriber2 in socks:
            subscriber2.recv_multipart()
            received += 1

    assert received == 2


def test_request_reply_routing(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    REQUEST messages are routed to registered service, REPLY routed back.
    """
    service = connected_client()

    register_msg = build_message(
        destination="calculator",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    service.send_multipart(register_msg)

    client = connected_client()

    register_client = build_message(
        destination="client-1",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    client.send_multipart(register_client)

    time.sleep(0.1)

    request_msg = build_message(
        destination="calculator",
        msg_type=MessageType.REQUEST,
        correlation_id="req-123",
        payload=b'{"operation": "add", "values": [1, 2]}',
    )
    client.send_multipart(request_msg)

    received, request_frames = wait_for_message(service, timeout_ms=2000)
    assert received
    assert request_frames is not None
    assert len(request_frames) == 4
    assert request_frames[1] == b"REQUEST"

    reply_msg = build_message(
        destination="client-1",
        msg_type=MessageType.REPLY,
        correlation_id="req-123",
        payload=b'{"result": 3}',
    )
    service.send_multipart(reply_msg)

    received, reply_frames = wait_for_message(client, timeout_ms=2000)
    assert received
    assert reply_frames is not None
    assert len(reply_frames) == 4
    assert reply_frames[1] == b"REPLY"
    assert reply_frames[2] == b"req-123"


def test_error_reply_for_unregistered_service(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Broker sends error REPLY when REQUEST targets unregistered service.
    """
    client = connected_client()

    register_msg = build_message(
        destination="client-1",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    client.send_multipart(register_msg)

    time.sleep(0.1)

    request_msg = build_message(
        destination="nonexistent-service",
        msg_type=MessageType.REQUEST,
        correlation_id="req-456",
        payload=b'{"query": "status"}',
    )
    client.send_multipart(request_msg)

    received, error_frames = wait_for_message(client, timeout_ms=2000)
    assert received
    assert error_frames is not None
    assert len(error_frames) == 4
    assert error_frames[1] == b"REPLY"
    assert error_frames[2] == b"req-456"
    assert b"error" in error_frames[3].lower()


def test_malformed_message_handling(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Broker handles malformed messages without crashing.
    """
    client = connected_client()

    client.send_multipart([b"only", b"two", b"frames"])

    time.sleep(0.2)
    assert started_proxy.is_running()


def test_multiple_services_independent_routing(
    started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket],
):
    """
    Multiple services can register and receive requests independently.
    """
    service_a = connected_client()
    register_a = build_message(
        destination="service-a",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    service_a.send_multipart(register_a)

    service_b = connected_client()
    register_b = build_message(
        destination="service-b",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    service_b.send_multipart(register_b)

    client = connected_client()

    time.sleep(0.1)
    assert started_proxy.get_service_count() == 2

    request_a = build_message(
        destination="service-a",
        msg_type=MessageType.REQUEST,
        correlation_id="req-a",
        payload=b"{}",
    )
    client.send_multipart(request_a)

    received_a, _ = wait_for_message(service_a, timeout_ms=2000)
    assert received_a

    received_b, _ = wait_for_message(service_b, timeout_ms=500)
    assert not received_b


def test_reply_to_unknown_correlation_id(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    REPLY with unknown correlation_id is logged and dropped without crashing.
    """
    service = connected_client()

    register_msg = build_message(
        destination="service-1",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    service.send_multipart(register_msg)

    time.sleep(0.1)

    reply_msg = build_message(
        destination="unknown-requester",
        msg_type=MessageType.REPLY,
        correlation_id="unknown-correlation-id",
        payload=b'{"result": "test"}',
    )
    service.send_multipart(reply_msg)

    time.sleep(0.2)
    assert started_proxy.is_running()


def test_service_reregistration_overwrites(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Re-registering same service_id overwrites previous registration.
    """
    service1 = connected_client()

    register_msg = build_message(
        destination="shared-service",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    service1.send_multipart(register_msg)

    time.sleep(0.1)
    assert started_proxy.get_service_count() == 1

    service2 = connected_client()
    service2.send_multipart(register_msg)

    time.sleep(0.1)
    assert started_proxy.get_service_count() == 1

    client = connected_client()

    request_msg = build_message(
        destination="shared-service",
        msg_type=MessageType.REQUEST,
        correlation_id="req-123",
        payload=b"{}",
    )
    client.send_multipart(request_msg)

    received_by_service1, _ = wait_for_message(service1, timeout_ms=500)
    received_by_service2, _ = wait_for_message(service2, timeout_ms=500)

    assert not received_by_service1
    assert received_by_service2


def test_parse_failure_invalid_utf8(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Invalid UTF-8 in message frames is handled gracefully.
    """
    client = connected_client()

    invalid_frames = [
        b"\xff\xfe\xfd",
        b"REQUEST",
        b"corr-123",
        b"payload",
    ]
    client.send_multipart(invalid_frames)

    time.sleep(0.2)
    assert started_proxy.is_running()


def test_parse_failure_invalid_message_type(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Invalid MessageType enum value is handled gracefully.
    """
    client = connected_client()

    invalid_frames = [
        b"destination",
        b"INVALID_MESSAGE_TYPE",
        b"corr-123",
        b"payload",
    ]
    client.send_multipart(invalid_frames)

    time.sleep(0.2)
    assert started_proxy.is_running()


def test_publish_to_topic_with_no_subscribers(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Publishing to topic with no subscribers completes without error.
    """
    publisher = connected_client()

    publish_msg = build_message(
        destination="empty-topic",
        msg_type=MessageType.PUBLISH,
        correlation_id="",
        payload=b'{"event": "test"}',
    )
    publisher.send_multipart(publish_msg)

    time.sleep(0.2)
    assert started_proxy.is_running()
    assert started_proxy.get_topic_subscriber_count("empty-topic") == 0


def test_socket_cleanup_after_stop(proxy: EventBusProxy, unique_port: int, zmq_context: zmq.Context):
    """
    Stop clears all registries and closes sockets.
    """
    proxy.start()
    time.sleep(0.1)

    service = zmq_context.socket(zmq.DEALER)
    service.connect(f"tcp://127.0.0.1:{unique_port}")

    register_msg = build_message(
        destination="test-service",
        msg_type=MessageType.REGISTER,
        correlation_id="",
        payload=b"{}",
    )
    service.send_multipart(register_msg)

    subscriber = zmq_context.socket(zmq.DEALER)
    subscriber.connect(f"tcp://127.0.0.1:{unique_port}")

    subscribe_msg = build_message(
        destination="test-topic",
        msg_type=MessageType.SUBSCRIBE,
        correlation_id="",
        payload=b"{}",
    )
    subscriber.send_multipart(subscribe_msg)

    time.sleep(0.1)
    assert proxy.get_service_count() == 1
    assert proxy.get_topic_subscriber_count("test-topic") == 1

    proxy.stop()

    assert proxy.get_service_count() == 0
    assert proxy.get_topic_subscriber_count("test-topic") == 0
    assert proxy.context is None
    assert proxy.router_socket is None

    service.close()
    subscriber.close()


def test_duplicate_subscriptions(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Same client subscribing to same topic multiple times results in single subscription.
    """
    subscriber = connected_client()

    subscribe_msg = build_message(
        destination="test-topic",
        msg_type=MessageType.SUBSCRIBE,
        correlation_id="",
        payload=b"{}",
    )
    subscriber.send_multipart(subscribe_msg)
    subscriber.send_multipart(subscribe_msg)
    subscriber.send_multipart(subscribe_msg)

    time.sleep(0.1)
    assert started_proxy.get_topic_subscriber_count("test-topic") == 1

    publisher = connected_client()

    publish_msg = build_message(
        destination="test-topic",
        msg_type=MessageType.PUBLISH,
        correlation_id="",
        payload=b'{"event": "test"}',
    )
    publisher.send_multipart(publish_msg)

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    received_count = 0
    deadline = time.time() + 1.0
    while time.time() < deadline:
        socks = dict(poller.poll(timeout=200))
        if subscriber in socks:
            subscriber.recv_multipart()
            received_count += 1

    assert received_count == 1


def test_unknown_message_type_handling(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Unknown message type is logged and ignored without crashing.
    """
    _ = connected_client()

    time.sleep(0.2)
    assert started_proxy.is_running()


def test_multiple_rapid_publishes(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Rapid succession of publishes are all delivered.
    """
    subscriber = connected_client()

    subscribe_msg = build_message(
        destination="rapid-topic",
        msg_type=MessageType.SUBSCRIBE,
        correlation_id="",
        payload=b"{}",
    )
    subscriber.send_multipart(subscribe_msg)

    time.sleep(0.1)

    publisher = connected_client()

    num_messages = 10
    for i in range(num_messages):
        publish_msg = build_message(
            destination="rapid-topic",
            msg_type=MessageType.PUBLISH,
            correlation_id="",
            payload=f'{{"seq": {i}}}'.encode(),
        )
        publisher.send_multipart(publish_msg)

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    received = 0
    deadline = time.time() + 2.0
    while received < num_messages and time.time() < deadline:
        socks = dict(poller.poll(timeout=500))
        if subscriber in socks:
            subscriber.recv_multipart()
            received += 1

    assert received == num_messages


def test_empty_topic_name(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Empty string topic name is handled like any other topic.
    """
    subscriber = connected_client()

    subscribe_msg = build_message(
        destination="",
        msg_type=MessageType.SUBSCRIBE,
        correlation_id="",
        payload=b"{}",
    )
    subscriber.send_multipart(subscribe_msg)

    time.sleep(0.1)
    assert started_proxy.get_topic_subscriber_count("") == 1

    publisher = connected_client()

    publish_msg = build_message(
        destination="",
        msg_type=MessageType.PUBLISH,
        correlation_id="",
        payload=b'{"event": "test"}',
    )
    publisher.send_multipart(publish_msg)

    received, _ = wait_for_message(subscriber, timeout_ms=1000)
    assert received


def test_long_topic_name(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Very long topic names are handled correctly.
    """
    long_topic = "a" * 1000

    subscriber = connected_client()

    subscribe_msg = build_message(
        destination=long_topic,
        msg_type=MessageType.SUBSCRIBE,
        correlation_id="",
        payload=b"{}",
    )
    subscriber.send_multipart(subscribe_msg)

    time.sleep(0.1)
    assert started_proxy.get_topic_subscriber_count(long_topic) == 1

    publisher = connected_client()

    publish_msg = build_message(
        destination=long_topic,
        msg_type=MessageType.PUBLISH,
        correlation_id="",
        payload=b'{"event": "test"}',
    )
    publisher.send_multipart(publish_msg)

    received, _ = wait_for_message(subscriber, timeout_ms=1000)
    assert received


def test_special_characters_in_topic(started_proxy: EventBusProxy, connected_client: Callable[[], zmq.Socket]):
    """
    Topics with special characters are handled correctly.
    """
    special_topic = "topic/with:special.chars-and_underscores"

    subscriber = connected_client()

    subscribe_msg = build_message(
        destination=special_topic,
        msg_type=MessageType.SUBSCRIBE,
        correlation_id="",
        payload=b"{}",
    )
    subscriber.send_multipart(subscribe_msg)

    time.sleep(0.1)
    assert started_proxy.get_topic_subscriber_count(special_topic) == 1

    publisher = connected_client()

    publish_msg = build_message(
        destination=special_topic,
        msg_type=MessageType.PUBLISH,
        correlation_id="",
        payload=b'{"event": "test"}',
    )
    publisher.send_multipart(publish_msg)

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    socks = dict(poller.poll(timeout=1000))
    assert subscriber in socks

    subscriber.close()
    publisher.close()
