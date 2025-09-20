"""Polling and waiting utilities for coredinator testing."""

import time
from collections.abc import Callable


def wait_for_event(pred: Callable[[], bool], interval: float, timeout: float) -> bool:
    """
    Wait for a predicate function to return True within a timeout period.

    Args:
        pred: Callable that returns True when condition is met
        interval: Time between checks in seconds
        timeout: Maximum time to wait in seconds

    Returns:
        True if condition was met, False if timeout occurred
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if pred():
            return True
        time.sleep(interval)
    return False
