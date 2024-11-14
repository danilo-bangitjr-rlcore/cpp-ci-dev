import logging
from datetime import UTC, datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from corerl.data_loaders.utils import try_connect
from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine

logger = logging.getLogger(__name__)


class DataReader:
    def __init__(self, db_cfg: SQLEngineConfig, db_name: str, sensor_table_name: str) -> None:
        self.engine: Engine = get_sql_engine(db_data=db_cfg, db_name=db_name)
        self.sensor_table_name = sensor_table_name
        self.connection = try_connect(self.engine)

    def batch_aggregated_read(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        bucket_width: timedelta,
        aggregation: str = "avg",
    ):
        query_str = f"""
            SELECT
                {_time_bucket(bucket_width=bucket_width, time_col='time')} as time_bucket,
                name,
                {_aggregator(aggregation=aggregation, val_col=_parse_jsonb('fields'), time_col='time')} AS val
            FROM {self.sensor_table_name}
            WHERE {_time_between('time', start_time, end_time)}
            AND {_filter_any('name', names)}
            GROUP BY time_bucket, name
            ORDER BY time_bucket ASC, name ASC;
        """

        sensor_data = pd.read_sql(sql=query_str, con=self.connection)
        if sensor_data.empty:
            logger.warning(f"failed query:\n{query_str}")
            raise Exception("dataframe returned from timescale was empty.")

        sensor_data = sensor_data.pivot(columns="name", values="val", index="time_bucket")

        missing_cols = set(names) - set(sensor_data.columns)
        sensor_data[list(missing_cols)] = np.nan

        return sensor_data

    def single_aggregated_read(
        self, names: List[str], start_time: datetime, end_time: datetime, aggregation: str = "avg"
    ):
        bucket_width = end_time - start_time

        query_str = f"""
            SELECT
                {_time_bucket(bucket_width=bucket_width, time_col='time', origin=start_time)} as time_bucket,
                name,
                {_aggregator(aggregation=aggregation, val_col=_parse_jsonb('fields'), time_col='time')} AS val
            FROM {self.sensor_table_name}
            WHERE {_time_between('time', start_time, end_time)}
            AND {_filter_any('name', names)}
            GROUP BY time_bucket, name
            ORDER BY time_bucket ASC, name ASC;
        """

        sensor_data = pd.read_sql(sql=query_str, con=self.connection)
        if sensor_data.empty:
            logger.warning(f"failed query:\n{query_str}")
            raise Exception("dataframe returned from timescale was empty.")

        # add time column to enable pivot
        sensor_data = sensor_data.pivot(columns="name", values="val", index="time_bucket")
        n_rows = sensor_data.shape[0]
        if n_rows != 1:
            logger.warning(
                f"single_aggregated_read returned {n_rows}, expected 1. Taking the last row (newest data)..."
            )
            sensor_data = sensor_data.iloc[-1]

        missing_cols = set(names) - set(sensor_data.columns)
        sensor_data[list(missing_cols)] = np.nan

        return sensor_data

    def close(self) -> None:
        self.connection.close()


def _time_bucket(bucket_width: timedelta, time_col: str, origin: datetime | None = None) -> str:
    """
    NOTE: Addition of bucket_width in the timebucket definition ensures that the labels
    for the time buckets align with the end of the bucket rather than the beginnning.
    """
    if origin is None:
        return f"time_bucket(INTERVAL '{bucket_width}', {time_col}) + '{bucket_width}'"
    else:
        assert origin.tzinfo == UTC
        origin_ts = f"TIMESTAMP '{origin.isoformat()}'"
        return f"time_bucket(INTERVAL '{bucket_width}', {time_col}, origin => {origin_ts}) + '{bucket_width}'"


def _aggregator(aggregation: str, val_col: str, time_col: str | None = None) -> str:
    assert aggregation in ["avg", "last"]
    if aggregation == "avg":
        return f"avg({val_col})"
    elif aggregation == "last":
        assert time_col is not None
        return f"last({val_col}, {time_col})"
    else:
        raise NotImplementedError


def _time_between(time_col: str, start: datetime, end: datetime) -> str:
    assert start.tzinfo == UTC
    assert end.tzinfo == UTC
    return f"""
        {time_col} > TIMESTAMP '{start.isoformat()}'
        AND {time_col} < TIMESTAMP '{end.isoformat()}'
    """


def _parse_jsonb(col: str, attribute: str = "val", type_str: str = "float") -> str:
    return f"({col}->'{attribute}')::{type_str}"


def _filter_any(col: str, vals: list[str]) -> str:
    s = "\tOR ".join([f"{col} = '{v}'\n" for v in vals])
    return f"({s})"
