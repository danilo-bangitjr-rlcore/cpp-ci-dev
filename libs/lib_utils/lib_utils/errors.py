import functools
import logging
from collections.abc import Callable
from typing import Any

from lib_utils.maybe import Maybe


def fail_gracefully[**P, R](logger: Any | None = None) -> Callable[[Callable[P, R]], Callable[P, Maybe[R]]]:
    if logger is None:
        logger = logging.getLogger()

    def decorator(f: Callable[P, R]):
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Maybe[R]:
            try:
                r = f(*args, **kwargs)
                return Maybe(r)
            except Exception as e:
                logger.exception(f"An error occurred in {f.__name__}: {e}")
                return Maybe[R](None)

        return wrapper
    return decorator
