import time
from collections.abc import Callable

import pytest


def wait_for_event(
    condition: Callable[[], bool],
    timeout: float = 30.0,
    interval: float = 0.5,
    description: str = "condition to be true",
) -> None:
    """
    Wait for a condition to become true within a timeout.
    """
    max_attempts = max(1, int(timeout / interval))

    for _ in range(max_attempts):
        if condition():
            return
        time.sleep(interval)

    pytest.fail(f"Timeout waiting for {description} after {timeout}s")
