import datetime as dt
import logging
from datetime import UTC, datetime, timedelta
from time import sleep
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

def clock_generator(tick_period: timedelta) -> Generator[datetime, None, None]:
    tick = datetime.now(UTC)
    tick.replace(microsecond=0) # trim microseconds
    while True:
        yield tick
        tick += tick_period

def wait_for_timestamp(timestamp: datetime) -> None:
    """
    Blocks until the requested timestamp
    """
    now = datetime.now(UTC)
    if now >= timestamp:
        sleep_duration = 0
    else:
        sleep_duration = (timestamp - now).total_seconds()
    sleep(sleep_duration)

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def split_windows_into_chunks(windows: list[tuple[dt.datetime | None, dt.datetime | None]], width: dt.timedelta):
    for start, stop in windows:
        if start is None or stop is None:
            logger.warning(f"Attempted to load historical window with missing datetime (start: {start}, stop:{stop})")
            continue

        logger.info(f"Offline chunks will be loaded from {start} to {stop}")
        yield from split_into_chunks(start, stop, width)

def split_into_chunks(start: dt.datetime, end: dt.datetime, width: dt.timedelta):
    s = start
    while s < end:
        e = s + width
        yield(s, min(end, e))
        s = e


def percent_time_elapsed(start: dt.datetime, end: dt.datetime, cur: dt.datetime | None = None) -> float:
    if cur is None:
        cur = dt.datetime.now(UTC)

    prop = (cur - start).total_seconds() / (end - start).total_seconds()
    return np.clip(prop, 0, 1)


# ----------------------
# -- Time conversions --
# ----------------------
Seconds = float
def as_seconds(s: float | dt.timedelta) -> float:
    if isinstance(s, dt.timedelta):
        return s.total_seconds()

    return s
