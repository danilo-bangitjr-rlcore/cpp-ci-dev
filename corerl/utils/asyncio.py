from collections.abc import Awaitable
from inspect import isawaitable

type MaybeAwaitable[T] = T | Awaitable[T]

async def maybe_await[T](t: MaybeAwaitable[T]) -> T:
    if isawaitable(t):
        return await t

    return t
