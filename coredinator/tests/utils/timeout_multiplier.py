import sys


def get_timeout_multiplier() -> float:
    if sys.platform.startswith('win'):
        return 3.0
    return 1.0


def apply_timeout_multiplier(timeout: float) -> float:
    return timeout * get_timeout_multiplier()


TIMEOUT_MULTIPLIER = get_timeout_multiplier()
