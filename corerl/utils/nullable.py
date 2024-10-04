from typing import Callable, TypeVar


T = TypeVar('T')
def default(thing: T | None, other: Callable[[], T]) -> T:
    if thing is None:
        return other()

    return thing
