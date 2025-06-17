import datetime as dt
from datetime import UTC


def now_iso() -> str:
    return dt.datetime.now(UTC).isoformat()

