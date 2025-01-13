import datetime as dt

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
