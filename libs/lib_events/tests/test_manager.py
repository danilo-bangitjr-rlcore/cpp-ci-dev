import socket
import time

import pytest

from lib_events.server.manager import EventBusManager


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
def manager(unique_ports: tuple[int, int]):
    """
    Create an EventBusManager instance with unique ports.
    """
    xsub_port, xpub_port = unique_ports
    mgr = EventBusManager(
        host="127.0.0.1",
        pub_port=xsub_port,
        sub_port=xpub_port,
    )
    yield mgr
    if mgr.is_healthy():
        mgr.stop()


@pytest.mark.timeout(5)
def test_manager_start_starts_proxy(manager: EventBusManager):
    """
    Starting manager starts underlying proxy.
    """
    assert not manager.is_healthy()

    manager.start()
    time.sleep(0.1)

    assert manager.is_healthy()
    assert manager.proxy.is_running()


@pytest.mark.timeout(5)
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


@pytest.mark.timeout(5)
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


@pytest.mark.timeout(5)
def test_manager_get_config_replaces_wildcard():
    """
    get_config replaces wildcard addresses with localhost.
    """
    manager = EventBusManager(
        host="*",
        pub_port=5559,
        sub_port=5560,
    )
    config = manager.get_config()

    assert config["publisher_endpoint"] == "tcp://localhost:5559"
    assert config["subscriber_endpoint"] == "tcp://localhost:5560"


@pytest.mark.timeout(5)
def test_manager_get_config_preserves_explicit_host():
    """
    get_config preserves explicit host addresses.
    """
    manager = EventBusManager(
        host="192.168.1.100",
        pub_port=5559,
        sub_port=5560,
    )
    config = manager.get_config()

    assert config["publisher_endpoint"] == "tcp://192.168.1.100:5559"
    assert config["subscriber_endpoint"] == "tcp://192.168.1.100:5560"


@pytest.mark.timeout(10)
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
