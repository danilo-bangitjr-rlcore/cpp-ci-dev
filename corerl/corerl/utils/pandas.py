import datetime
from datetime import UTC

import pandas as pd


def get_datetime(df: pd.DataFrame, col: str, idx: int) -> datetime.datetime:
    x = df[col][idx]
    assert isinstance(x, pd.Timestamp)
    return x.to_pydatetime().astimezone(UTC)
