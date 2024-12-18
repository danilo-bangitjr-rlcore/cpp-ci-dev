import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, List, Literal, assert_never

import numpy as np
import pandas as pd
from sqlalchemy import TEXT, TIMESTAMP, Column, Engine, Float, MetaData, Table, cast, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import text

import corerl.utils.pandas as pd_util
from corerl.configs.config import config
from corerl.data_pipeline.db.utils import try_connect
from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine

logger = logging.getLogger(__name__)

@config()
class TagDBConfig(SQLEngineConfig):
    """Configuration to setup the tag database connection.
    """
    db_name: str = "postgres"
    sensor_table_name: str = "scrubber4"
    sensor_table_schema: str = "public"


class DataReader:
    def __init__(self, db_cfg: TagDBConfig) -> None:
        self.engine: Engine = get_sql_engine(db_data=db_cfg, db_name=db_cfg.db_name)
        self.connection = try_connect(self.engine)

        self.db_metadata = MetaData()
        self.db_metadata.reflect(bind=self.engine)
        self.sensor_table = Table(
            db_cfg.sensor_table_name,
            self.db_metadata,
            Column("time", TIMESTAMP, nullable=False, default= None, autoincrement= False, comment=None),
            Column("host", TEXT, nullable=True, default= None, autoincrement= False, comment="tag"),
            Column("id", TEXT, nullable=True, default= None, autoincrement= False, comment="tag"),
            Column("name", TEXT, nullable=True, default= None, autoincrement= False, comment="tag"),
            Column("Quality", TEXT, nullable=True, default= None, autoincrement= False, comment="tag"),
            Column("fields", JSONB, nullable=True, default= None, autoincrement= False, comment=None),
            schema=db_cfg.sensor_table_schema,
            extend_existing=True
        )

    def batch_aggregated_read(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        bucket_width: timedelta,
        aggregation: Literal["avg"] | Literal["last"] = "avg",
    ):
        # https://docs.timescale.com/api/latest/hyperfunctions/time_bucket/#time_bucket
        time_bucket_stmt = func.time_bucket(
            text(f"INTERVAL '{bucket_width}'"),
            self.sensor_table.c["time"],
            text(f"origin => '{start_time.isoformat()}'"),
            text("timezone => 'UTC'")
        )

        match(aggregation):
            case "avg":
                # https://www.postgresql.org/docs/17/functions-aggregate.html
                agg_stmt = func.avg(cast(self.sensor_table.c["fields"]["val"], Float))
            case "last":
                # https://docs.timescale.com/api/latest/hyperfunctions/last/#last
                agg_stmt = func.last(self.sensor_table.c["fields"]["val"], self.sensor_table.c["time"])
            case _:
                assert_never(aggregation)

        stmt = select(
            time_bucket_stmt.label("time_bucket"),
            self.sensor_table.c["name"],
            agg_stmt.label("val")
        ).filter(
            self.sensor_table.c["time"] >= text(f"TIMESTAMP '{start_time.isoformat()}'"),
            self.sensor_table.c["time"] < text(f"TIMESTAMP '{end_time.isoformat()}'"),
            self.sensor_table.c["name"].in_(names)
        ).group_by(
            text("time_bucket"), self.sensor_table.c["name"]
        ).order_by(
            text("time_bucket ASC"), self.sensor_table.c["name"].asc()
        )

        logger.debug(stmt.compile(self.engine, compile_kwargs={"literal_binds": True}))

        sensor_data = pd.read_sql(sql=stmt, con=self.connection)
        sensor_data = sensor_data.pivot(columns="name", values="val", index="time_bucket")

        missing_cols = set(names) - set(sensor_data.columns)
        sensor_data[list(missing_cols)] = np.nan

        t = start_time
        while t < end_time:
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
        return self.batch_aggregated_read(names, start_time, end_time, bucket_width, aggregation)

    def close(self) -> None:
        self.connection.close()

    def query(self, q: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = params or {}

        q = q.replace(":table", self.sensor_table.name)
        q = q.replace(":val", _parse_jsonb("fields"))

        return pd.read_sql(
            sql=text(q),
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
        df = self.query(q, { "tag": tag_name })
        return TagStats(
            tag=tag_name,
            min=df["min"].item(),
            max=df["max"].item(),
            avg=df["avg"].item(),
            var=df["var"].item(),
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
            start=pd_util.get_datetime(df, "start", 0),
            end=pd_util.get_datetime(df, "end", 0),
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


def _parse_jsonb(col: str, attribute: str = "val", type_str: str = "float") -> str:
    return f"({col}->'{attribute}')::{type_str}"


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
