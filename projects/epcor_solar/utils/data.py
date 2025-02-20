import datetime as dt
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd
from cloudpathlib import CloudPath, S3Client, S3Path
from cloudpathlib.enums import FileCacheMode


def _split_columns(df: pd.DataFrame) -> list[pd.Series]:
    dfs = []
    for column_name in df.columns:
        dfs.append(df[column_name])

    return dfs

def _totals_to_deltas(df: pd.Series) -> pd.Series:
    """
    Some tags track running totals.
    This method transforms the column to track deltas between consecutive rows instead
    """
    delta_df = df.diff()
    delta_df = delta_df.drop(delta_df.index[0])

    return delta_df


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
    # In the electricity price data, numbers have commas to separate "thousands" (Eg: 3,456 instead of 3456)
    df = df.replace(',', '', regex=True)
    df = df.astype('float32')

    return df


def _adjust_pool_price_forecast_timestamps(series: pd.Series) -> pd.Series:
    """
    The forecast listed at a given timestamp was the forecast for that hour.
    The forecast was published in the preceding hour so we should adjust its timestamp accordingly
    """
    delta = dt.timedelta(hours=1)
    series.index = series.index - delta

    return series


def _parse_pool_price_data(df: pd.DataFrame) -> list[pd.Series]:
    """
    Parse electricity pool price data obtained from AESO: http://ets.aeso.ca/
    """
    df = _parse_pool_price_timestamps(df)
    columns = _split_columns(df)
    transformed_columns = []
    for column in columns:
        assert isinstance(column.name, str)
        if "Forecast" in column.name:
            column = _adjust_pool_price_forecast_timestamps(column)

        transformed_columns.append(column)

    return transformed_columns


def _parse_battery_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Battery cycle data is only logged once a day at 1AM
    - Some timestamps only contain the date. Others also include the time
    """
    df[['Date', 'Hour', 'AM/PM']] = df['Timestamp'].str.split(' ', expand=True)
    df['Hour'] = "1:00:00"
    df['Timestamp'] = df[['Date', 'Hour']].agg(' '.join, axis=1)
    df = df.drop(['Date', 'Hour', 'AM/PM'], axis=1)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("Timestamp")
    df = df.replace(',', '', regex=True)
    df = df.astype('float32')

    return df


def _parse_battery_data(df: pd.DataFrame) -> list[pd.Series]:
    """
    Parse battery cycle data for BESS 1A/B and BESS 2A/B
    """
    df = _parse_battery_timestamps(df)
    columns = _split_columns(df)
    transformed_columns = []
    for column in columns:
        assert isinstance(column.name, str)
        if "CYCLE" in column.name:
            column = _totals_to_deltas(column)

        transformed_columns.append(column)

    return transformed_columns


def _parse_solar_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%m/%d/%y %H:%M")
    except ValueError:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%m/%d/%Y %H:%M")
    df = df.set_index("Timestamp")
    df = df.replace(',', '', regex=True)
    df = df.astype('float32')

    return df


def _parse_solar_data(df: pd.DataFrame) -> list[pd.Series]:
    """
    Parse data in Solar_Data_A.csv or Solar_Data_B.csv
    """
    df = _parse_solar_timestamps(df)
    columns = _split_columns(df)
    transformed_columns = []
    for column in columns:
        column_name = column.name
        column_name = cast(str, column_name)
        if "_TOT" in column_name:
            column = _totals_to_deltas(column)
        if column_name[-4:] == "_KVA":
            column = _abs(column)

        transformed_columns.append(column)

    return transformed_columns

def _parse_setpoint_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format="%m/%d/%Y %H:%M")
    df = df.set_index("Timestamp")
    df = df.replace(',', '', regex=True)
    df = df.astype('float32')

    return df

def _parse_setpoint_data(df: pd.DataFrame) -> list[pd.Series]:
    """
    Parse data in Setpoints.csv
    """
    df = _parse_setpoint_timestamps(df)
    columns = _split_columns(df)
    transformed_columns = []
    for column in columns:
        assert isinstance(column.name, str)
        if "_TOT" in column.name:
            column = _totals_to_deltas(column)

        transformed_columns.append(column)

    return transformed_columns

def load_csv_files(files: Iterable[CloudPath]) -> list[pd.Series]:
    columns = []
    for file in files:
        df = pd.read_csv(file)
        df = df.rename(columns={df.columns[0]: "Timestamp"})
        if "Pool_Price" in str(file):
            pool_price_columns = _parse_pool_price_data(df)
            columns += pool_price_columns
        elif "battery" in str(file):
            battery_columns = _parse_battery_data(df)
            columns += battery_columns
        elif "Solar_Data" in str(file):
            solar_columns = _parse_solar_data(df)
            columns += solar_columns
        elif "Setpoints" in str(file):
            setpoint_columns = _parse_setpoint_data(df)
            columns += setpoint_columns

    return columns

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
