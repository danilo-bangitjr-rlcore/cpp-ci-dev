from omegaconf import DictConfig
from datetime import datetime, timedelta, UTC
from corerl.sql_logging.sql_logging import get_sql_engine
import pandas as pd
from sqlalchemy import Engine
from typing import List, Any, Union
import numpy as np
from corerl.data_loaders.utils import try_connect
import logging

logger = logging.getLogger(__name__)


class DataReader:
    def __init__(self, db_cfg: DictConfig, db_name: str, sensor_table_name: str) -> None:
        db_data = dict(db_cfg)
        self.engine: Engine = get_sql_engine(db_data=db_data, db_name=db_name)
        self.sensor_table_name = sensor_table_name
        self.connection = try_connect(self.engine)

    def batch_aggregated_read(
        self, names: List[str], start_time: datetime, end_time: datetime, bucket_width: timedelta
    ):
        """
        NOTE: Addition of bucket_width in the select statement ensures that the labels
        for the time buckets align with the end of the bucket rather than the beginnning.
        """
        
        query_str = f"""
            SELECT 
              time_bucket(INTERVAL '{bucket_width}', time) + '{bucket_width}' as time_bucket,
              name,
              avg({_parse_jsonb('fields')}) AS avg_val
            FROM {self.sensor_table_name}
            WHERE {_time_between('time', start_time, end_time)}
            AND {_filter_any('name', names)}
            GROUP BY time_bucket, name
            ORDER BY time_bucket DESC, name ASC;
        """

        sensor_data = pd.read_sql(sql=query_str, con=self.connection)
        if sensor_data.empty:
            logger.warning(f"failed query:\n{query_str}")
            raise Exception("dataframe returned from timescale was empty.")

        sensor_data = sensor_data.pivot(columns="name", values="avg_val", index="time_bucket")

        missing_cols = set(names) - set(sensor_data.columns)
        sensor_data[list(missing_cols)] = np.nan

        return sensor_data

    def single_aggregated_read(self, names: List[str], start_time: datetime, end_time: datetime):
        
        query_str = f"""
            SELECT 
              name,
              avg({_parse_jsonb('fields')}) AS avg_val
            FROM {self.sensor_table_name}
            WHERE {_time_between('time', start_time, end_time)}
            AND {_filter_any('name', names)}
            GROUP BY name
            ORDER BY name ASC;
        """

        sensor_data = pd.read_sql(sql=query_str, con=self.connection)
        if sensor_data.empty:
            logger.warning(f"failed query:\n{query_str}")
            raise Exception("dataframe returned from timescale was empty.")

        # add time column to enable pivot
        sensor_data["time"] = end_time
        sensor_data = sensor_data.pivot(columns="name", values="avg_val", index="time")
        
        missing_cols = set(names) - set(sensor_data.columns)
        sensor_data[list(missing_cols)] = np.nan

        return sensor_data

    def close(self) -> None:
        self.connection.close()

def _time_between(time_col: str, start: datetime, end: datetime) -> str:
    assert start.tzinfo == UTC
    assert end.tzinfo == UTC
    return f"""
        {time_col} > TIMESTAMP '{start.isoformat()}'
        AND {time_col} < TIMESTAMP '{end.isoformat()}'
    """

def _parse_jsonb(col: str, attribute: str = 'val', type_str: str = 'float') -> str:
    return f"({col}->'{attribute}')::{type_str}"

def _filter_any(col: str, vals: list[str]) -> str:
    s = '\tOR '.join([f"{col} = '{v}'\n" for v in vals])
    return f'({s})'


def fill_data_for_changed_setpoint(
        change_tags: List[str],
        dfs: List[pd.DataFrame],
        delta_t: timedelta
) -> List[tuple[datetime, str, Any]]:
    data_tuples: List[tuple[datetime, str, Any]] = []
    largest_timestamp = None
    for df in dfs:
        columns = df.columns.values.tolist()  # datetime, tag, value
        if df.iloc[0, 1] not in change_tags:
            largest_timestamp = df[columns[0]].max() \
                if largest_timestamp is None else \
                max(largest_timestamp, df[columns[0]].max())

    for df in dfs:
        if df.iloc[0, 1] in change_tags:
            for idx in range(len(df)-1):
                data_tuples += _fillin_between(df, idx, delta_t)
            # use largest_timestamp+delta_t to take care of the timestamp on the edge
            assert largest_timestamp is not None
            data_tuples += _fillin_between(df, len(df) - 1, delta_t, largest_timestamp+delta_t)
        else:
            data_tuples += list(zip(*map(df.get, df)))
    return data_tuples

def _fillin_between(df: pd.DataFrame, row: int, delta_t: timedelta, end_ts: datetime | None=None) \
        -> List[tuple[datetime, str, Any]]:
    start_ts, tag, value = df.iloc[row]
    if end_ts is None:
        end_ts, _, _ = df.iloc[row+1]
    ts = start_ts
    tuples: List[tuple[datetime, str, Any]] = []
    while ts < end_ts:
        tuples.append((ts, tag, value))
        ts += delta_t
    return tuples
