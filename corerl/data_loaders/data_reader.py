from datetime import datetime, timedelta, UTC
from corerl.sql_logging.sql_logging import get_sql_engine, SQLEngineConfig
import pandas as pd
from sqlalchemy import Engine
from typing import List
import numpy as np
from corerl.data_loaders.utils import try_connect
import logging

logger = logging.getLogger(__name__)


class DataReader:
    def __init__(self, db_cfg: SQLEngineConfig, db_name: str, sensor_table_name: str) -> None:
        self.engine: Engine = get_sql_engine(db_data=db_cfg, db_name=db_name)
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
