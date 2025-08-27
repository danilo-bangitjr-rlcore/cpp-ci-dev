import datetime as dt
import logging
from collections.abc import Generator, Iterable
from datetime import UTC, datetime, timedelta
from time import sleep

import numpy as np

logger = logging.getLogger(__name__)

def clock_generator(tick_period: timedelta) -> Generator[datetime]:
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

def split_windows_into_chunks(windows: list[tuple[dt.datetime | None, dt.datetime | None]], width: dt.timedelta):
    for start, stop in windows:
        if start is None or stop is None:
            logger.warning(f"Attempted to load historical window with missing datetime (start: {start}, stop:{stop})")
            continue

        logger.info(f"Offline chunks will be loaded from {start} to {stop}")
        yield from split_into_chunks(start, stop, width)

def split_into_chunks(start: dt.datetime, end: dt.datetime, width: dt.timedelta):
    assert (
        start < end
    ), "The start timestamp passed to split_into_chunks() must come before the passed end timestamp."

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

def exclude_from_chunks(
        time_chunks: Iterable[tuple[dt.datetime, dt.datetime]],
        exclude_chunks: Iterable[tuple[dt.datetime, dt.datetime]],
    ):
    filtered_chunks = []
    for time_chunk in time_chunks:
        filtered_chunks.extend(exclude_from_chunk(time_chunk, exclude_chunks))

    return filtered_chunks

def exclude_from_chunk(
        time_chunk: tuple[dt.datetime, dt.datetime],
        exclude_chunks: Iterable[tuple[dt.datetime, dt.datetime]],
    ):
    current_chunks = [time_chunk]

    for exclude_chunk in exclude_chunks:
        new_chunks = []
        for current_chunk in current_chunks:
            excluded_chunks = exclude_from_chunk_single(current_chunk, exclude_chunk)
            new_chunks.extend(excluded_chunks)
        current_chunks = new_chunks

    return current_chunks

def exclude_from_chunk_single(
        time_chunk: tuple[dt.datetime, dt.datetime],
        exclude_chunk: tuple[dt.datetime, dt.datetime],
    ):

    return_chunks = []
    curr_start, curr_end = time_chunk[0], time_chunk[1]
    exclude_start, exclude_end = exclude_chunk[0], exclude_chunk[1]

    if exclude_end <= curr_start or curr_end <= exclude_start:
        # No overlap
        return_chunks = [(curr_start, curr_end)]

    elif curr_start < exclude_start and curr_end > exclude_end:
        # Evaluation period is inside chunk - split into two
        return_chunks.append((curr_start, exclude_start))
        return_chunks.append((exclude_end, curr_end))

    elif curr_start < exclude_start:
        # Partial overlap at end
        return_chunks.append((curr_start, exclude_start))

    elif curr_end > exclude_end:
        # Partial overlap at start
        return_chunks.append((exclude_end, curr_end))

    # else if chunk is completely inside eval period, don't add anything
    return return_chunks


# ----------------------
# -- Time conversions --
# ----------------------
Seconds = float
def as_seconds(s: float | dt.timedelta) -> float:
    if isinstance(s, dt.timedelta):
        return s.total_seconds()

    return s
