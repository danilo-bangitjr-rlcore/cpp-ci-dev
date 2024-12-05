import datetime as dt

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def split_into_chunks(start: dt.datetime, end: dt.datetime, width: dt.timedelta):
    s = start
    sec = dt.timedelta(seconds=1)
    while s < end:
        e = s + width
        # ensure the last bucket does not reach past end
        e_exclusive = min(end, e - sec)
        yield(s, e_exclusive)
        s = e


# ----------------------
# -- Time conversions --
# ----------------------
Seconds = float
def as_seconds(s: float | dt.timedelta) -> float:
    if isinstance(s, dt.timedelta):
        return s.total_seconds()

    return s
