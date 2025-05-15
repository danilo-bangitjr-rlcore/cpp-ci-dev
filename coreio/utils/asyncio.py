import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from datetime import UTC, datetime, timedelta
from inspect import isawaitable

logger = logging.getLogger(__name__)


type MaybeAwaitable[T] = T | Awaitable[T]

async def maybe_await[T](t: MaybeAwaitable[T]) -> T:
    if isawaitable(t):
        return await t

    return t

def exponential_backoff(
    exceptions: Iterable[type[Exception]] = (Exception,),
    attempts: int = 10,
):
    def _inner[T, **P](f: Callable[P, Coroutine[None, None, T]]):
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            retries = 0
            last_error = datetime.now(UTC)
            while retries < attempts:
                try:
                    return await f(*args, **kwargs)

                except tuple(exceptions) as err:
                    retries += 1

                    # if we have gone long enough without an error
                    # reset the counter
                    now = datetime.now(UTC)
                    if now - last_error > timedelta(hours=1):
                        retries = 1

                    # backoff exponentially over seconds
                    last_error = now
                    sec = 2 ** (retries - 1)
                    logger.warning(f"{err}")
                    logger.warning(f"Trying again in {sec} seconds")
                    await asyncio.sleep(sec)

            raise Exception(f'Failed after {attempts} attempts')

        return wrapper
    return _inner
