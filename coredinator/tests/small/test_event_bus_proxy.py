import socket
import time

import pytest
import zmq

from coredinator.event_bus.proxy import EventBusProxy


@pytest.fixture
def unique_ports() -> tuple[int, int]:
    """
    Generate unique port numbers for each test to avoid conflicts.
    """
    def get_free_port():
        sock = socket.socket()
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    xsub_port = get_free_port()
    xpub_port = get_free_port()
    return xsub_port, xpub_port


@pytest.fixture
def proxy(unique_ports: tuple[int, int]):
    """
    Create an EventBusProxy instance with unique ports.
    """
    xsub_port, xpub_port = unique_ports
    proxy = EventBusProxy(
        xsub_addr=f"tcp://127.0.0.1:{xsub_port}",
        xpub_addr=f"tcp://127.0.0.1:{xpub_port}",
    )
    yield proxy
    if proxy.is_running():
        proxy.stop()


@pytest.mark.timeout(5)
def test_proxy_stop_terminates_thread(proxy: EventBusProxy):
    """
    Stopping proxy terminates thread and cleans up resources.
    """
    proxy.start()
    time.sleep(0.1)
    assert proxy.is_running()

    proxy.stop()

    assert not proxy.is_running()


@pytest.mark.timeout(5)
def test_proxy_idempotent_stop(proxy: EventBusProxy):
    """
    Stopping an already stopped proxy does nothing.
    """
    proxy.stop()
    proxy.stop()
    assert not proxy.is_running()


@pytest.mark.timeout(10)
def test_proxy_forwards_messages_from_publisher_to_subscriber(proxy: EventBusProxy, unique_ports: tuple[int, int]):
    """
    Messages published to XSUB are forwarded to subscribed XPUB clients.
    """
    xsub_port, xpub_port = unique_ports
    proxy.start()
    time.sleep(0.2)

    context = zmq.Context()

    publisher = context.socket(zmq.PUB)
    publisher.connect(f"tcp://127.0.0.1:{xsub_port}")

    subscriber = context.socket(zmq.SUB)
    subscriber.connect(f"tcp://127.0.0.1:{xpub_port}")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "test.topic")

    time.sleep(0.3)

    publisher.send_multipart([b"test.topic", b"test message"])

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    socks = dict(poller.poll(timeout=2000))

    assert subscriber in socks
    topic, message = subscriber.recv_multipart()
    assert topic == b"test.topic"
    assert message == b"test message"

    publisher.close()
    subscriber.close()
    context.term()


@pytest.mark.timeout(10)
def test_proxy_topic_filtering(proxy: EventBusProxy, unique_ports: tuple[int, int]):
    """
    Subscriber only receives messages matching subscribed topics.
    """
    xsub_port, xpub_port = unique_ports
    proxy.start()
    time.sleep(0.2)

    context = zmq.Context()

    publisher = context.socket(zmq.PUB)
    publisher.connect(f"tcp://127.0.0.1:{xsub_port}")

    subscriber = context.socket(zmq.SUB)
    subscriber.connect(f"tcp://127.0.0.1:{xpub_port}")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "allowed")

    time.sleep(0.3)

    publisher.send_multipart([b"allowed.event", b"should receive"])
    publisher.send_multipart([b"blocked.event", b"should not receive"])

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    socks = dict(poller.poll(timeout=2000))
    assert subscriber in socks
    topic, message = subscriber.recv_multipart()
    assert topic == b"allowed.event"
    assert message == b"should receive"

    socks = dict(poller.poll(timeout=500))
    assert subscriber not in socks

    publisher.close()
    subscriber.close()
    context.term()


@pytest.mark.timeout(10)
def test_proxy_multiple_subscribers(proxy: EventBusProxy, unique_ports: tuple[int, int]):
    """
    Multiple subscribers receive the same message.
    """
    xsub_port, xpub_port = unique_ports
    proxy.start()
    time.sleep(0.3)

    context = zmq.Context()

    subscriber1 = context.socket(zmq.SUB)
    subscriber1.connect(f"tcp://127.0.0.1:{xpub_port}")
    subscriber1.setsockopt_string(zmq.SUBSCRIBE, "")

    subscriber2 = context.socket(zmq.SUB)
    subscriber2.connect(f"tcp://127.0.0.1:{xpub_port}")
    subscriber2.setsockopt_string(zmq.SUBSCRIBE, "")

    time.sleep(0.5)

    publisher = context.socket(zmq.PUB)
    publisher.connect(f"tcp://127.0.0.1:{xsub_port}")

    time.sleep(0.5)

    publisher.send_multipart([b"broadcast", b"message for all"])

    received_count = 0
    messages = []
    poller = zmq.Poller()
    poller.register(subscriber1, zmq.POLLIN)
    poller.register(subscriber2, zmq.POLLIN)

    deadline = time.time() + 3.0
    while received_count < 2 and time.time() < deadline:
        socks = dict(poller.poll(timeout=500))
        if subscriber1 in socks:
            messages.append(("sub1", subscriber1.recv_multipart()))
            received_count += 1
        if subscriber2 in socks:
            messages.append(("sub2", subscriber2.recv_multipart()))
            received_count += 1

    assert received_count == 2, f"Expected 2 messages, got {received_count}"
    assert len(messages) == 2

    for _, (topic, msg) in messages:
        assert topic == b"broadcast"
        assert msg == b"message for all"

    publisher.close()
    subscriber1.close()
    subscriber2.close()
    context.term()


@pytest.mark.timeout(5)
def test_proxy_survives_stop_event_during_poll_error(proxy: EventBusProxy):
    """
    Proxy handles ZMQError gracefully during polling when stop is signaled.
    """
    proxy.start()
    time.sleep(0.1)
    assert proxy.is_running()

    proxy.stop()
    time.sleep(0.1)

    assert not proxy.is_running()
