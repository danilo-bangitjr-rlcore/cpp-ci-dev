import functools
import logging
from collections.abc import Callable

from lib_utils.maybe import Maybe


def fail_gracefully[**P, R](logger: logging.Logger | None = None) -> Callable[[Callable[P, R]], Callable[P, Maybe[R]]]:
    if logger is None:
        logger = logging.getLogger()

    def decorator(f: Callable[P, R]):
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Maybe[R]:
            try:
                r = f(*args, **kwargs)
                return Maybe(r)
            except Exception as e:
                logger.exception("An error occurred in %s: %s", f.__name__, e)
                return Maybe[R](None)

        return wrapper
    return decorator
