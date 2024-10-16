import datetime as dt

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()



# ----------------------
# -- Time conversions --
# ----------------------
Seconds = float
def as_seconds(s: float | dt.timedelta) -> float:
    if isinstance(s, dt.timedelta):
        return s.total_seconds()

    return s
