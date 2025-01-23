import datetime as dt
from datetime import UTC, datetime, timedelta
from time import sleep
from typing import Generator


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


def split_into_chunks(start: dt.datetime, end: dt.datetime, width: dt.timedelta):
    s = start
    while s < end:
        e = s + width
        yield(s, min(end, e))
        s = e


# ----------------------
# -- Time conversions --
# ----------------------
Seconds = float
def as_seconds(s: float | dt.timedelta) -> float:
    if isinstance(s, dt.timedelta):
        return s.total_seconds()

    return s
