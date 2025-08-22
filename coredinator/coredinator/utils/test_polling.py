"""Test utilities for polling and waiting."""

import time
from collections.abc import Callable


def wait_for_event(pred: Callable[[], bool], interval: float, timeout: float) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        if pred():
            return True
        time.sleep(interval)
    return False
