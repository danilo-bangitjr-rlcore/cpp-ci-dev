from typing import Callable


def default[T](thing: T | None, other: Callable[[], T]) -> T:
    if thing is None:
        return other()

    return thing
