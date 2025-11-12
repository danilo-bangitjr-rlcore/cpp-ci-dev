import socket
import time

import pytest

from lib_events.server.manager import EventBusManager


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
def manager(unique_port: int):
    """
    Create an EventBusManager instance with unique port.
    """
    mgr = EventBusManager(host="127.0.0.1", port=unique_port)
    yield mgr
    if mgr.is_healthy():
        mgr.stop()


def test_manager_start_starts_proxy(manager: EventBusManager):
    """
    Starting manager starts underlying proxy.
    """
    assert not manager.is_healthy()

    manager.start()
    time.sleep(0.1)

    assert manager.is_healthy()
    assert manager.proxy.is_running()


def test_manager_stop_stops_proxy(manager: EventBusManager):
    """
    Stopping manager stops underlying proxy.
    """
    manager.start()
    time.sleep(0.1)
    assert manager.is_healthy()

    manager.stop()

    assert not manager.is_healthy()
    assert not manager.proxy.is_running()


def test_manager_health_reflects_proxy_state(manager: EventBusManager):
    """
    Manager health check reflects proxy running state.
    """
    assert not manager.is_healthy()

    manager.start()
    time.sleep(0.1)
    assert manager.is_healthy()

    manager.stop()
    assert not manager.is_healthy()


def test_manager_get_config_replaces_wildcard():
    """
    get_config replaces wildcard addresses with localhost.
    """
    manager = EventBusManager(host="*", port=5580)
    config = manager.get_config()

    assert config["endpoint"] == "tcp://localhost:5580"


def test_manager_get_config_preserves_explicit_host():
    """
    get_config preserves explicit host addresses.
    """
    manager = EventBusManager(host="192.168.1.100", port=5580)
    config = manager.get_config()

    assert config["endpoint"] == "tcp://192.168.1.100:5580"


def test_manager_get_service_count(manager: EventBusManager):
    """
    Manager exposes service count from broker.
    """
    manager.start()
    time.sleep(0.1)

    assert manager.get_service_count() == 0


def test_manager_get_topic_subscriber_count(manager: EventBusManager):
    """
    Manager exposes topic subscriber count from broker.
    """
    manager.start()
    time.sleep(0.1)

    assert manager.get_topic_subscriber_count("test-topic") == 0


def test_manager_lifecycle_multiple_cycles(manager: EventBusManager):
    """
    Manager can be started and stopped multiple times.
    """
    for _ in range(3):
        manager.start()
        time.sleep(0.1)
        assert manager.is_healthy()

        manager.stop()
        assert not manager.is_healthy()
