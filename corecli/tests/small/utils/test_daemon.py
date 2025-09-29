import subprocess
import time
from collections.abc import Generator

import psutil
import pytest

from corecli.utils.daemon import stop_process_gracefully, wait_for_event
from tests.small.utils.utils import cleanup_process, start_test_daemon


class TestWaitForEvent:
    def test_wait_for_event_success(self) -> None:
        call_count = 0

        def predicate() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 3

        result = wait_for_event(predicate, 0.01, 1.0)
        assert result is True
        assert call_count >= 3

    def test_wait_for_event_timeout(self) -> None:
        def predicate() -> bool:
            return False

        start_time = time.time()
        result = wait_for_event(predicate, 0.01, 0.1)
        elapsed = time.time() - start_time

        assert result is False
        assert elapsed >= 0.1

    def test_wait_for_event_immediate_success(self) -> None:
        def predicate() -> bool:
            return True

        result = wait_for_event(predicate, 0.01, 1.0)
        assert result is True


# Pytest fixtures
@pytest.fixture
def daemon_process() -> Generator[subprocess.Popen[bytes]]:
    """Provide a normal test daemon process."""
    proc = start_test_daemon("normal")
    assert wait_for_event(lambda: psutil.pid_exists(proc.pid), 0.05, 2.0)
    yield proc
    cleanup_process(proc)


@pytest.fixture
def daemon_ignore_sigterm() -> Generator[subprocess.Popen[bytes]]:
    """Provide a test daemon process that ignores SIGTERM."""
    proc = start_test_daemon("ignore_sigterm")
    assert wait_for_event(lambda: psutil.pid_exists(proc.pid), 0.05, 2.0)
    yield proc
    cleanup_process(proc)


class TestStopProcessGracefully:
    def test_stop_process_success(self, daemon_process: subprocess.Popen[bytes]) -> None:
        pid = daemon_process.pid
        assert psutil.pid_exists(pid)
        result = stop_process_gracefully(pid, timeout=2.0)
        assert result is True
        assert not psutil.pid_exists(pid)

    def test_stop_nonexistent_process(self) -> None:
        fake_pid = 999999
        assert not psutil.pid_exists(fake_pid)
        result = stop_process_gracefully(fake_pid)
        assert result is True

    def test_stop_process_requires_force(self, daemon_ignore_sigterm: subprocess.Popen[bytes]) -> None:
        pid = daemon_ignore_sigterm.pid
        assert psutil.pid_exists(pid)
        result = stop_process_gracefully(pid, timeout=1.0)
        assert result is True
        assert not psutil.pid_exists(pid)
