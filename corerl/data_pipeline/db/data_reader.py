import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, List, Literal, assert_never

import numpy as np
import pandas as pd
from sqlalchemy import Engine
import sqlalchemy

import corerl.utils.pandas as pd_util
from corerl.configs.config import config, MISSING
from corerl.data_pipeline.db.utils import try_connect
from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine

logger = logging.getLogger(__name__)

@config()
class TagDBConfig(SQLEngineConfig):
    db_name: str = MISSING
    sensor_table_name: str = MISSING


class DataReader:
    def __init__(self, db_cfg: TagDBConfig) -> None:
        self.engine: Engine = get_sql_engine(db_data=db_cfg, db_name=db_cfg.db_name)
        self.sensor_table_name = db_cfg.sensor_table_name
        self.connection = try_connect(self.engine)

    def batch_aggregated_read(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        bucket_width: timedelta,
        aggregation: Literal["avg"] | Literal["last"] = "avg",
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

        t = start_time
        while t <= end_time:
            if t not in sensor_data.index:
                idx = pd.DatetimeIndex([t])

                # type erasure because pandas...
                cols: Any = names
                row = pd.DataFrame([[np.nan] * len(names)], columns=cols, index=idx)
                sensor_data = pd.concat((sensor_data, row), axis=0, copy=False)

            t += bucket_width

        return sensor_data.sort_index()

    def single_aggregated_read(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        aggregation: Literal["avg"] | Literal["last"] = "avg",
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


    def query(self, q: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = params or {}

        q = q.replace(':table', self.sensor_table_name)
        q = q.replace(':val', _parse_jsonb('fields'))

        return pd.read_sql(
            sql=sqlalchemy.text(q),
            con=self.connection,
            params=params,
        )

    def get_tag_stats(self, tag_name: str):
        q = """
            SELECT
              MIN(:val) as min,
              MAX(:val) as max,
              AVG(:val) as avg,
              VARIANCE(:val) as var
            FROM :table
            WHERE name=:tag
        """
        df = self.query(q, { 'tag': tag_name })
        return TagStats(
            tag=tag_name,
            min=df['min'].item(),
            max=df['max'].item(),
            avg=df['avg'].item(),
            var=df['var'].item(),
        )

    def get_time_stats(self):
        q = """
            SELECT
              MIN(time) as start,
              MAX(time) as end
            FROM :table
        """
        df = self.query(q)
        return TimeStats(
            start=pd_util.get_datetime(df, 'start', 0),
            end=pd_util.get_datetime(df, 'end', 0),
        )


@dataclass
class TimeStats:
    start: datetime
    end: datetime

@dataclass
class TagStats:
    tag: str
    min: float | None
    max: float | None
    avg: float | None
    var: float | None


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


def _aggregator(aggregation: Literal["avg"] | Literal["last"], val_col: str, time_col: str | None = None) -> str:
    match aggregation:
        case "avg":
            return f"avg({val_col})"
        case "last":
            assert time_col is not None
            return f"last({val_col}, {time_col})"
        case _:
            assert_never(aggregation)


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


def fill_data_for_changed_setpoint(
    change_tags: List[str], dfs: List[pd.DataFrame], delta_t: timedelta
) -> List[tuple[datetime, str, Any]]:
    data_tuples: List[tuple[datetime, str, Any]] = []
    largest_timestamp = None
    for df in dfs:
        columns = df.columns.values.tolist()  # datetime, tag, value
        tag = df.iloc[0, 1]
        if tag not in change_tags:
            largest_timestamp = (
                df[columns[0]].max() if largest_timestamp is None else max(largest_timestamp, df[columns[0]].max())
            )

    for df in dfs:
        if df.iloc[0, 1] in change_tags:
            for idx in range(len(df) - 1):
                data_tuples += _fillin_between(df, idx, delta_t)
            # use largest_timestamp+delta_t to take care of the timestamp on the edge
            assert largest_timestamp is not None
            data_tuples += _fillin_between(df, len(df) - 1, delta_t, largest_timestamp + delta_t)
        else:
            data_tuples += list(zip(*map(df.get, df), strict=True))
    return data_tuples


def _fillin_between(
    df: pd.DataFrame, row: int, delta_t: timedelta, end_ts: datetime | None = None
) -> List[tuple[datetime, str, Any]]:
    start_ts, tag, value = df.iloc[row]
    if end_ts is None:
        end_ts, _, _ = df.iloc[row + 1]
    ts = start_ts
    tuples: List[tuple[datetime, str, Any]] = []
    while ts < end_ts:
        tuples.append((ts, tag, value))
        ts += delta_t
    return tuples
