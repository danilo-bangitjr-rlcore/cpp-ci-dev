from collections.abc import Awaitable
from inspect import isawaitable
from typing import TypeVar

T = TypeVar('T')
MaybeAwaitable = T | Awaitable[T]

async def maybe_await(t: MaybeAwaitable[T]) -> T:
    if isawaitable(t):
        return await t

    return t
