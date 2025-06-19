import datetime as dt
import numpy as np
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pandas as pd
from cloudpathlib import CloudPath, S3Client, S3Path
from cloudpathlib.enums import FileCacheMode
from load_data import SolarDataLoaderConfig


def _split_columns(df: pd.DataFrame) -> list[pd.Series]:
    return [
        df[column_name]
        for column_name in df.columns
    ]

def _abs(series: pd.Series) -> pd.Series:
    return abs(series)

def _parse_pool_price_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    - AESO's timestamps follow a weird convention where they count hours in the day from 1-24 rather than from 0-23.
    - Daylight savings is indicated by an asterisk *
    - They place commas after every third digit (i.e 1,000 instead of 1000)
    """
    df[['Date', 'Hour']] = df['Timestamp'].str.split(' ', expand=True)
    df['Hour'] = df['Hour'].str.rstrip("*")
    df['Hour'] = df['Hour'].astype('int32')
    df['Hour'] = df['Hour'] - 1
    df['Hour'] = df['Hour'].astype('str')
    df['Timestamp'] = df[['Date', 'Hour']].agg(' '.join, axis=1)
    df = df.drop(['Date', 'Hour'], axis=1)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%m/%d/%Y %H")
    df = df.set_index("Timestamp")
    df = df[~df.index.duplicated(keep='first')]
    # In the electricity price data, numbers have commas to separate "thousands" (Eg: 3,456 instead of 3456)
    df = df.replace(',', '', regex=True)
    return df.astype('float64')

def _adjust_pool_price_forecast_timestamps(series: pd.Series, offset: int) -> pd.Series:
    """
    The forecast listed at a given timestamp was the forecast for that hour.
    The forecast was published in the preceding hour so we should adjust its timestamp accordingly
    """
    delta = dt.timedelta(minutes=offset)
    series.index = series.index - delta

    return series

def _rename_pool_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("Forecast AIL & Actual AIL Difference", axis=1)
    df = df.rename(
        columns={
            "Forecast Pool Price": "Forecast_Pool_Price_1",
            "Actual Posted Pool Price": "Actual_Posted_Pool_Price",
            "Forecast AIL": "Forecast_AIL_1",
            "Actual AIL": "Current_AIL",
        }
    )

    return df

def _get_forecast(
    data: pd.Series,
    inference_window: int,
    obs_period: dt.timedelta,
    std_init: float,
    std_mult: float
) -> list[tuple[Any, ...]]:
    """
    The historical pool price and AIL data only contains the last forecast
    that was made 5 minutes before the start of the given hour.
    This function creates forecasts for the next three hours by sampling values
    from a normal distribution whose mean was the last forecast that was made for the given hour.
    The standard deviation of the distribution increases as the forecast horizon increases.
    """
    sql_tups = []
    for timestamp, value in data.items():
        assert isinstance(timestamp, dt.datetime)
        offset = dt.timedelta(minutes=0)
        std = value * std_init
        while offset < dt.timedelta(hours=inference_window):
            curr_time = timestamp - offset
            forecast_hour = int(offset / dt.timedelta(hours=1)) + 1
            tag_name = data.name[:-1] + str(forecast_hour)
            if curr_time == timestamp:
                sql_tups.append((curr_time, value, tag_name))
            else:
                sampled_value = np.random.normal(loc=value, scale=std)
                sampled_value = np.clip(sampled_value, a_min=0, a_max=None)
                sql_tups.append((curr_time, sampled_value, tag_name))

            std *= std_mult
            offset += obs_period

    return sql_tups

def _get_observed_values(
    data: pd.Series,
    inference_window: int,
    obs_period: dt.timedelta
) -> list[tuple[Any, ...]]:
    """
    The reported pool price and AIL for a given hour is constant (doesn't fluctuate within the hour).
    This function ensures these values are logged at the obs_period frequency within the sensors db.
    """
    sql_tups = []
    for timestamp, value in data.items():
        assert isinstance(timestamp, dt.datetime)
        offset = dt.timedelta(minutes=0)
        while offset < dt.timedelta(hours=inference_window):
            curr_time = timestamp + offset
            tag_name = data.name
            sql_tups.append((curr_time, value, tag_name))
            offset += obs_period

    return sql_tups

def _parse_pool_price_data(cfg: SolarDataLoaderConfig, df: pd.DataFrame) -> list[tuple[Any, ...]]:
    """
    Parse electricity pool price data obtained from AESO: http://ets.aeso.ca/
    """
    df = _parse_pool_price_timestamps(df)
    df = _rename_pool_price_columns(df)
    columns = _split_columns(df)
    sql_tups = []
    for column in columns:
        assert isinstance(column.name, str)
        if "Forecast" in column.name:
            column = _adjust_pool_price_forecast_timestamps(column, 5)
            sql_tups += _get_forecast(
                data=column,
                inference_window=3,
                obs_period=cfg.obs_period,
                std_init=cfg.std_init,
                std_mult=cfg.std_mult,
            )
        else:
            sql_tups += _get_observed_values(
                data=column,
                inference_window=1,
                obs_period=cfg.obs_period,
            )

    return sql_tups

def _parse_solar_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%m/%d/%y %H:%M")
    except ValueError:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%m/%d/%Y %H:%M")
    df = df.set_index("Timestamp")
    df = df.replace(',', '', regex=True)
    return df.astype('float64')

def _remove_all_bess_offline_entries(cfg: SolarDataLoaderConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove entries where neither BESS 1 nor BESS 2 are online (actions have no effect)
    """
    if "SCADA1.ELS_SLR_PS1000_1A_AVAIL.A_CV" in df.columns:
        bess_avails = []
        if cfg.bess1:
            bess_avails += ["SCADA1.ELS_SLR_PS1000_1A_AVAIL.A_CV", "SCADA1.ELS_SLR_PS1000_1B_AVAIL.A_CV"]
        if cfg.bess2:
            bess_avails += ["SCADA1.ELS_SLR_PS1000_2A_AVAIL.A_CV", "SCADA1.ELS_SLR_PS1000_2B_AVAIL.A_CV"]

        df = df.loc[~(df[bess_avails] == 0).all(axis=1)]

    return df

def _parse_solar_data(cfg: SolarDataLoaderConfig, df: pd.DataFrame) -> list[tuple[Any, ...]]:
    """
    Parse data in Solar_Data_A.csv, Solar_Data_B.csv, or Solar_Data_C.csv
    """
    df = _parse_solar_timestamps(df)
    df = _remove_all_bess_offline_entries(cfg, df)
    columns = _split_columns(df)
    transformed_columns = []
    for column in columns:
        column_name = column.name
        column_name = cast(str, column_name)
        if column_name[-4:] == "_KVA":
            column = _abs(column)

        transformed_columns.append(column)

    return columns_to_sql_tups(transformed_columns)

def load_csv_files(cfg: SolarDataLoaderConfig, files: Iterable[CloudPath]) -> list[tuple[Any, ...]]:
    sql_tups = []
    for file in files:
        df = pd.read_csv(file)
        df = df.rename(columns={df.columns[0]: "Timestamp"})
        if "Pool_Price" in str(file):
            sql_tups += _parse_pool_price_data(cfg, df)
        elif "Solar_Data" in str(file):
            sql_tups += _parse_solar_data(cfg, df)

    return sql_tups

def columns_to_sql_tups(columns: list[pd.Series]) -> list[tuple[Any, ...]]:
    """
    Converting pd.Series into (timestamp, tag_name, value) tuples
    """
    sql_tups = []
    for column in columns:
        tag = column.name
        df = column.to_frame()
        df["Tag"] = [tag] * len(df)
        sql_tups += list(df.itertuples(index=True, name=None))

    return sql_tups

def get_s3_files() -> Iterable[CloudPath]:
    cache = Path('projects/epcor_solar/.cache')
    cache.mkdir(parents=True, exist_ok=True)
    client = S3Client(
        file_cache_mode=FileCacheMode.persistent,
        local_cache_dir=cache,
    )
    root = S3Path('s3://rlcore-shared/', client)
    return root.glob('epcor_solar/*.csv')
