import datetime as dt
from collections import namedtuple
from pathlib import Path
from typing import Iterable, cast

import pandas as pd
from cloudpathlib import CloudPath, S3Client, S3Path
from cloudpathlib.enums import FileCacheMode

from corerl.corerl.configs.config import config, list_

SQLEntry = namedtuple('SQLEntry', ['time', 'val', 'tag'])

@config()
class VictoriaWWConfig:
    setpoint_change_tags: list[str] = list_()
    default_tag_freq: dt.timedelta = dt.timedelta(minutes=15)


def _split_columns(df: pd.DataFrame) -> list[pd.Series]:
    series_list = []
    for column_name in df.columns:
        series_list.append(df[column_name])

    return series_list

def get_last_timestamp(series_list: list[pd.Series]) -> dt.datetime:
    """
    Some tags only have readings at setpoint changes.
    Need to know the last global timestamp in the offline data
    to be able to carry the last readings forward for these tags.
    """
    global_last_timestamp = pd.Timestamp(year=1, month=1, day=1, hour=1, tzinfo=dt.UTC)
    for series in series_list:
        series_last_timestamp = series.index[-1]
        assert isinstance(series_last_timestamp, dt.datetime)
        if series_last_timestamp > global_last_timestamp:
            global_last_timestamp = series_last_timestamp

    return global_last_timestamp

def _copy_tag_val_between_timestamps(
    start: dt.datetime,
    end: dt.datetime,
    step: dt.timedelta,
    tag_name: str,
    value: float
) -> list[SQLEntry]:
    sql_tups = []
    curr_time = start
    while curr_time < end:
        sql_tups.append(SQLEntry(curr_time, float(value), tag_name))
        curr_time += step

    return sql_tups

def _parse_setpoint_change_data(
    dl_cfg: VictoriaWWConfig,
    column: pd.Series,
    final_timestamp: dt.datetime
) -> list[SQLEntry]:
    """
    Parse tags that only have entries at setpoint changes
    """
    sql_tups = []
    current_idx = 0
    next_idx = 1
    column_len = len(column)
    tag_name = str(column.name)

    while next_idx < column_len:
        current_timestamp = column.index[current_idx]
        current_timestamp = cast(dt.datetime, current_timestamp)
        curr_val = column.iloc[current_idx]
        next_timestamp = column.index[next_idx]
        next_timestamp = cast(dt.datetime, next_timestamp)
        sql_tups += _copy_tag_val_between_timestamps(current_timestamp,
                                                     next_timestamp,
                                                     dl_cfg.default_tag_freq,
                                                     tag_name,
                                                     curr_val)
        current_idx += 1
        next_idx += 1

    # Fill in missing sql tups between last timestamp in series and global 'final_timestamp'
    current_timestamp = column.index[current_idx]
    current_timestamp = cast(dt.datetime, current_timestamp)
    curr_val = column.iloc[current_idx]
    sql_tups += _copy_tag_val_between_timestamps(current_timestamp,
                                                 final_timestamp,
                                                 dl_cfg.default_tag_freq,
                                                 tag_name,
                                                 curr_val)

    return  sql_tups

def _series_to_sql_tups(
    series: pd.Series
) -> list[SQLEntry]:
    """
    Converting pd.Series into (timestamp, tag_name, value) tuples
    """
    tag = series.name
    df = series.to_frame()
    df["Tag"] = [tag] * len(df)
    sql_tups = list(map(lambda tup: SQLEntry(*tup), df.itertuples(index=True, name=None)))

    return sql_tups

def get_sql_tups(
    dl_cfg: VictoriaWWConfig,
    series_list: list[pd.Series],
    last_timestamp: dt.datetime
) -> list[SQLEntry]:
    sql_tups = []
    for series in series_list:
        tag_name = series.name
        series = series.dropna()

        if tag_name in dl_cfg.setpoint_change_tags:
            sql_tups += _parse_setpoint_change_data(dl_cfg, series, last_timestamp)
        else:
            sql_tups += _series_to_sql_tups(series)

    return sql_tups

def load_excel_files(files: Iterable[CloudPath]) -> list[pd.Series]:
    series_list = []
    for file in files:
        df = pd.read_excel(file)
        df = df.rename(columns={df.columns[0]: "Timestamp"})
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%m/%d/%Y %H:%M")
        except ValueError:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%m/%d/%y %H:%M")
        df = df.set_index("Timestamp")
        df = df.astype('float32')

        series_list += _split_columns(df)

    return series_list

def get_s3_files() -> Iterable[CloudPath]:
    cache = Path('projects/victoria_ww/.cache')
    cache.mkdir(parents=True, exist_ok=True)
    client = S3Client(
        file_cache_mode=FileCacheMode.persistent,
        local_cache_dir=cache,
    )
    root = S3Path('s3://rlcore-shared/', client)
    return root.glob('victoria_ww/*.xlsx')
