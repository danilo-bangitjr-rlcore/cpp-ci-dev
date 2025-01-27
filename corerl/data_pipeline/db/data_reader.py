import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import groupby
from typing import Any, List, Literal, assert_never

import pandas as pd
from sqlalchemy import TEXT, TIMESTAMP, Boolean, Column, Engine, Float, MetaData, Table, cast, func, select, union_all
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import text

import corerl.utils.pandas as pd_util
from corerl.data_pipeline.db.data_writer import TagDBConfig
from corerl.data_pipeline.db.utils import TryConnectContextManager
from corerl.sql_logging.sql_logging import get_sql_engine

logger = logging.getLogger(__name__)


class DataReader:
    def __init__(self, db_cfg: TagDBConfig) -> None:
        self.engine: Engine = get_sql_engine(db_data=db_cfg, db_name=db_cfg.db_name)

        self.db_metadata = MetaData()
        self.db_metadata.reflect(bind=self.engine)
        self.sensor_table = Table(
            db_cfg.table_name,
            self.db_metadata,
            Column("time", TIMESTAMP, nullable=False, default=None, autoincrement=False, comment=None),
            Column("host", TEXT, nullable=True, default=None, autoincrement=False, comment="tag"),
            Column("id", TEXT, nullable=True, default=None, autoincrement=False, comment="tag"),
            Column("name", TEXT, nullable=True, default=None, autoincrement=False, comment="tag"),
            Column("fields", JSONB, nullable=True, default=None, autoincrement=False, comment=None),
            schema=db_cfg.table_schema,
            extend_existing=True,
        )

    def batch_aggregated_read(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        bucket_width: timedelta,
        aggregation: Literal["avg", "last", "bool_or"] = "avg",
        tag_aggregations: dict[str, Literal["avg", "last", "bool_or"]] | None = None,
    ) -> pd.DataFrame:
        """
        The intended behavior is for buckets to be inclusive wrt their end_time and exclusive wrt their start_time.
        The bucket start/end times are distinct concepts from the read start_time and end_time.

        The inclusion/exclusion of data on the bucket boundaries is handled by timescale and
        not configurable (to our knowledge). The default behevior of timescale's time_bucket
        function is to include data at the start of the bucket time interval, and
        exclude data at the end of the bucket time interval.

        If we denote the start/end of the bucket time interval as b_start/b_end respectively, the bucket includes data
        with timestamps in the half open interval [b_start, b_end). Note that none of our code handles bucket time
        intervals directly; this is handled by timescale at query time.

        To align with our notion that observations occur at a point t in time, we instead would like to construct
        observations at time t that are inclusive of data with timestamp t, but exclusive of data
        with timestamp t - obs_period. That is, we want the opposite of the default timescale bucketing behavior:
        We want data with timestamps in the half open interval (t-obs_period, t]. We also want the row in the
        resultant dataframe to be labeled with a timestamp index t.

        Note: typically obs_period == bucket_width

        This function corrects for the discrepancy between the default timescale behavior and our desired behavior.
        Please see the tests in the TestDataReaderLogic class in test/medium/data_loaders/test_data_reader.py
        for concrete examples of input and expected output.
        """
        # If timezone unaware, assume we are using naive system timezone and convert to UTC.
        if start_time.tzinfo is None:
            start_tz = start_time.astimezone().tzinfo
            logger.warning(f"naive start_time passed, assuming {start_tz} and converting to UTC")
            start_time = start_time.replace(tzinfo=start_tz).astimezone(UTC)
        if end_time.tzinfo is None:
            end_tz = end_time.astimezone().tzinfo
            logger.warning(f"naive end_time passed, assuming {end_tz} and converting to UTC")
            end_time = end_time.replace(tzinfo=end_tz).astimezone(UTC)

        # https://docs.timescale.com/api/latest/hyperfunctions/time_bucket/#time_bucket
        # an additional microsecond is added to the origin to capture data at the bucket
        # end time and exclude data at the bucket start time.
        time_bucket_stmt = func.time_bucket(
            text(f"INTERVAL '{bucket_width}'"),
            self.sensor_table.c["time"],
            text(f"origin => '{(end_time+timedelta(microseconds=1)).isoformat()}'"),
            text("timezone => 'UTC'"),
        )
        # change the timestamp index of each row to align with the bucket end time
        # also correct for the microsecond that was added to the origin
        time_bucket_stmt += text(f"'{bucket_width-timedelta(microseconds=1)}'")

        # if tag_aggregations is not provided, use the default aggregation for all tags
        # if a tag is not in tag_aggregations, use the default aggregation for that tag
        tag_aggregations = tag_aggregations or {}
        aggregation_map = {name: tag_aggregations.get(name, aggregation) for name in names}

        agg_groups = {}
        sorted_names = sorted(names, key=lambda x: str(aggregation_map[x]))
        for agg_type, group in groupby(sorted_names, key=lambda x: aggregation_map[x]):
            agg_groups[agg_type] = list(group)

        subqueries = []
        for agg_type, tags in agg_groups.items():
            match agg_type:
                case "avg":
                    # https://www.postgresql.org/docs/17/functions-aggregate.html
                    agg_stmt = func.cast(
                        func.avg(cast(self.sensor_table.c["fields"]["val"], Float)),
                        TEXT
                    )
                case "last":
                    # https://docs.timescale.com/api/latest/hyperfunctions/last/#last
                    agg_stmt = func.cast(
                        func.last(self.sensor_table.c["fields"]["val"], self.sensor_table.c["time"]),
                        TEXT
                    )
                case "bool_or":
                    # needed to support truncated/terminated booleans
                    agg_stmt = func.cast(
                        func.bool_or(cast(self.sensor_table.c["fields"]["val"], Boolean)),
                        TEXT
                    )
                case _:
                    assert_never(agg_type)

            subquery = (
                select(
                    time_bucket_stmt.label("time_bucket"),
                    self.sensor_table.c["name"],
                    agg_stmt.label("val")
                )
                .filter(
                    self.sensor_table.c["time"] > text(f"TIMESTAMP '{start_time.isoformat()}'"),
                    self.sensor_table.c["time"] <= text(f"TIMESTAMP '{end_time.isoformat()}'"),
                    self.sensor_table.c["name"].in_(tags),
                )
                .group_by(text("time_bucket"), self.sensor_table.c["name"])
            )
            subqueries.append(subquery)

        # after executing the query and getting the df, pivot and convert values
        stmt = union_all(*subqueries).order_by(text("time_bucket ASC"), self.sensor_table.c["name"].asc())
        logger.debug(stmt.compile(self.engine, compile_kwargs={"literal_binds": True}))
        with TryConnectContextManager(self.engine) as connection:
            sensor_data = pd.read_sql(sql=stmt, con=connection)
            full_range = pd.date_range(
                start=start_time + bucket_width,
                end=end_time,
                freq=bucket_width,
                tz='UTC',
                name=None
            )

            if not sensor_data.empty:
                sensor_data = sensor_data.pivot(index='time_bucket', columns='name', values='val')
                missing_cols = set(names) - set(sensor_data.columns)
                for col in missing_cols:
                    sensor_data[col] = pd.NA
                sensor_data = sensor_data.reindex(full_range)
            else:
                sensor_data = pd.DataFrame(index=full_range, columns=pd.Index(names))

            for col in sensor_data.columns:
                col_agg = aggregation_map.get(col)
                if col_agg == "avg" or col_agg == "last":
                    sensor_data[col] = pd.to_numeric(sensor_data[col])
                elif col_agg == "bool_or":
                    sensor_data[col] = sensor_data[col].replace({"true": True, "false": False})

        return sensor_data

    def single_aggregated_read(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        aggregation: dict[str, Literal["avg", "last", "bool_or"]] | Literal["avg", "last", "bool_or"] = "avg",
    ) -> pd.DataFrame:
        bucket_width = end_time - start_time
        df = self.batch_aggregated_read(
            names,
            start_time,
            end_time,
            bucket_width,
            "avg" if isinstance(aggregation, dict) else aggregation,
            tag_aggregations=aggregation if isinstance(aggregation, dict) else None
        )

        if len(df) > 1:
            logger.warning("single_aggregated_read returned multiple rows. Only the last row will be returned")

        return pd.DataFrame(df[-1:])

    def close(self) -> None:
        return

    def query(self, q: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = params or {}

        q = q.replace(":table", self.sensor_table.name)
        q = q.replace(":val", _parse_jsonb("fields"))

        with TryConnectContextManager(self.engine) as connection:
            sensor_data = pd.read_sql(
                sql=text(q),
                con=connection,
                params=params,
            )

        return sensor_data

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
        df = self.query(q, {"tag": tag_name})
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
