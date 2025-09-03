import datetime
from datetime import UTC

import pandas as pd


def get_datetime(df: pd.DataFrame, col: str, idx: int) -> datetime.datetime:
    x = df[col][idx]
    assert isinstance(x, pd.Timestamp)
    return x.to_pydatetime().astimezone(UTC)

def split_dataframe_into_chunks(df: pd.DataFrame, chunk_length: int):
    """Split a pandas DataFrame into chunks of specified length"""
    chunks = []
    for i in range(0, len(df), chunk_length):
        chunk = df.iloc[i:i + chunk_length]
        chunks.append(chunk)
    return chunks
