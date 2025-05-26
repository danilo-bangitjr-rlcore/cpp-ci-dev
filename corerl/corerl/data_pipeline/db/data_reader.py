import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import groupby
from math import floor
from typing import Any, List, assert_never

import pandas as pd
from sqlalchemy import TEXT, TIMESTAMP, Boolean, Column, Float, MetaData, Table, cast, func, select, union_all
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import text

import corerl.utils.pandas as pd_util
from corerl.data_pipeline.db.data_writer import TagDBConfig
from corerl.data_pipeline.db.utils import TryConnectContextManager
from corerl.data_pipeline.tag_config import Agg
from corerl.sql_logging.sql_logging import get_sql_engine

logger = logging.getLogger(__name__)


class DataReader:
    def __init__(self, db_cfg: TagDBConfig) -> None:
        self.wide_format = db_cfg.wide_format
        self.engine = get_sql_engine(db_data=db_cfg, db_name=db_cfg.db_name)

        self.db_metadata = MetaData()
        self.db_metadata.reflect(bind=self.engine)

        if not self.wide_format:
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
        else:
            self.sensor_table = Table(
                db_cfg.table_name,
                self.db_metadata,
                Column("time", TIMESTAMP, nullable=False, primary_key=True),
                schema=db_cfg.table_schema,
                extend_existing=True
            )

    def batch_aggregated_read(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        bucket_width: timedelta,
        aggregation: Agg = Agg.avg,
        tag_aggregations: dict[str, Agg] | None = None,
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
        if start_time.tzinfo is None:
            start_tz = start_time.astimezone().tzinfo
            logger.warning(f"naive start_time passed, assuming {start_tz} and converting to UTC")
            start_time = start_time.replace(tzinfo=start_tz).astimezone(UTC)
        if end_time.tzinfo is None:
            end_tz = end_time.astimezone().tzinfo
            logger.warning(f"naive end_time passed, assuming {end_tz} and converting to UTC")
            end_time = end_time.replace(tzinfo=end_tz).astimezone(UTC)

        if self.wide_format:
            if tag_aggregations is None:
                raise ValueError("tag_aggregations is required for wide format")
            return self._batch_aggregated_read_wide(
                names,
                start_time,
                end_time,
                bucket_width,
                tag_aggregations
            )
        else:
            return self._batch_aggregated_read_narrow(
                names,
                start_time,
                end_time,
                bucket_width,
                aggregation,
                tag_aggregations
            )

    def _batch_aggregated_read_wide(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        bucket_width: timedelta,
        tag_aggregations: dict[str, Agg],
    ) -> pd.DataFrame:
        bucket_width_str = f"{bucket_width.total_seconds()} seconds"

        table_schema = self.sensor_table.schema
        table_name = self.sensor_table.name

        column_types = {}
        with TryConnectContextManager(self.engine) as connection:
            for name in names:
                try:
                    type_query = f"""
                        SELECT data_type
                        FROM information_schema.columns
                        WHERE table_schema = '{table_schema}'
                        AND table_name = '{table_name}'
                        AND column_name = '{name}'
                    """
                    result = connection.execute(text(type_query)).fetchone()
                    if not result:
                        raise ValueError(f"Column '{name}' does not exist in the database")
                    column_types[name] = result[0].lower()
                except Exception as e:
                    if isinstance(e, ValueError):
                        raise
                    raise ValueError(f"Error checking column '{name}': {e}") from e

        # add 1 microsecond to the origin to match the narrow format behavior
        # align with bucket end time by adding bucket_width - 1 microsecond
        select_parts = [
            f"time_bucket(INTERVAL '{bucket_width_str}', time, "
            f"origin => '{(end_time+timedelta(microseconds=1)).isoformat()}'::timestamptz, "
            f"timezone => 'UTC') + INTERVAL '{bucket_width_str}' - INTERVAL '1 microsecond' AS time_bucket"
        ]

        for name in names:
            agg_type = tag_aggregations[name]
            column_type = column_types.get(name, "unknown")

            if column_type == "boolean":
                if agg_type == Agg.last:
                    select_parts.append(f'last("{name}", time) as "{name}"')
                elif agg_type == Agg.bool_or:
                    select_parts.append(f'bool_or("{name}") as "{name}"')
                else:
                    raise ValueError(f"Unsupported aggregation {agg_type} for boolean column {name}")
            else:
                if agg_type == Agg.avg:
                    select_parts.append(f'avg("{name}") as "{name}"')
                elif agg_type == Agg.last:
                    select_parts.append(f'last("{name}", time) as "{name}"')
                elif agg_type == Agg.bool_or:
                    select_parts.append(f'bool_or("{name}"::boolean) as "{name}"')
                else:
                    raise ValueError(f"Unsupported aggregation {agg_type} for column {name}")

        query = f"""
            SELECT {', '.join(select_parts)}
            FROM {table_schema}.{table_name}
            WHERE time > '{start_time.isoformat()}'::timestamptz
            AND time <= '{end_time.isoformat()}'::timestamptz
            GROUP BY time_bucket
            ORDER BY time_bucket ASC
        """

        n_buckets = floor((end_time - start_time).total_seconds() / bucket_width.total_seconds())

        full_range = pd.date_range(
            start=end_time - (n_buckets - 1) * bucket_width,
            end=end_time,
            freq=bucket_width,
            tz='UTC',
            name=None
        )

        with TryConnectContextManager(self.engine) as connection:
            try:
                sensor_data = pd.read_sql(sql=text(query), con=connection)

                if not sensor_data.empty:
                    sensor_data.set_index('time_bucket', inplace=True)
                    sensor_data = sensor_data.reindex(full_range)
                else:
                    sensor_data = pd.DataFrame(index=full_range, columns=pd.Index(names))

                for col in sensor_data.columns:
                    col_agg = tag_aggregations.get(col)
                    col_type = column_types.get(col, "unknown")

                    if col_type == "boolean" or col_agg == Agg.bool_or:
                        sensor_data[col] = sensor_data[col].astype(bool)
                    else:
                        sensor_data[col] = pd.to_numeric(sensor_data[col])

            except Exception as e:
                logger.error(f"Error in wide format read: {e}")
                sensor_data = pd.DataFrame(index=full_range, columns=pd.Index(names))

        return sensor_data

    def _batch_aggregated_read_narrow(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        bucket_width: timedelta,
        aggregation: Agg = Agg.avg,
        tag_aggregations: dict[str, Agg] | None = None,
    ) -> pd.DataFrame:
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
                case Agg.avg:
                    # https://www.postgresql.org/docs/17/functions-aggregate.html
                    agg_stmt = func.cast(
                        func.avg(cast(self.sensor_table.c["fields"]["val"], Float)),
                        TEXT
                    )
                case Agg.last:
                    # https://docs.timescale.com/api/latest/hyperfunctions/last/#last
                    agg_stmt = func.cast(
                        func.last(self.sensor_table.c["fields"]["val"], self.sensor_table.c["time"]),
                        TEXT
                    )
                case Agg.bool_or:
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
                    self.sensor_table.c["time"] > text(f"TIMESTAMP WITH TIME ZONE '{start_time.isoformat()}'"),
                    self.sensor_table.c["time"] <= text(f"TIMESTAMP WITH TIME ZONE '{end_time.isoformat()}'"),
                    self.sensor_table.c["name"].in_(tags),
                )
                .group_by(text("time_bucket"), self.sensor_table.c["name"])
            )
            subqueries.append(subquery)

        # after executing the query and getting the df, pivot and convert values
        stmt = union_all(*subqueries).order_by(text("time_bucket ASC"), self.sensor_table.c["name"].asc())
        logger.debug(stmt.compile(self.engine, compile_kwargs={"literal_binds": True}))
        n_buckets = floor((end_time - start_time).total_seconds() / bucket_width.total_seconds())
        with TryConnectContextManager(self.engine) as connection:
            sensor_data = pd.read_sql(sql=stmt, con=connection)
            full_range = pd.date_range(
                start=end_time - (n_buckets - 1) * bucket_width,
                # start=start_time + bucket_width,
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
                if col_agg in {'avg', 'last'}:
                    sensor_data[col] = pd.to_numeric(sensor_data[col])
                elif col_agg == "bool_or":
                    sensor_data[col] = sensor_data[col].replace({"true": True, "false": False})

        return sensor_data

    def single_aggregated_read(
        self,
        names: List[str],
        start_time: datetime,
        end_time: datetime,
        aggregation: Agg = Agg.avg,
        tag_aggregations: dict[str, Agg] | None = None,
    ) -> pd.DataFrame:
        bucket_width = end_time - start_time
        df = self.batch_aggregated_read(
            names,
            start_time,
            end_time,
            bucket_width,
            aggregation,
            tag_aggregations
        )

        if len(df) > 1:
            logger.warning("single_aggregated_read returned multiple rows. Only the last row will be returned")

        return pd.DataFrame(df[-1:])

    def close(self) -> None:
        return

    def query(self, q: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = params or {}

        q = q.replace(":table", self.sensor_table.name)

        if not self.wide_format:
            q = q.replace(":val", _parse_jsonb("fields"))

        with TryConnectContextManager(self.engine) as connection:
            sensor_data = pd.read_sql(
                sql=text(q),
                con=connection,
                params=params,
            )

        return sensor_data

    def get_tag_stats(self, tag_name: str):
        if self.wide_format:
            with TryConnectContextManager(self.engine) as connection:
                check_query = f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = '{self.sensor_table.schema}'
                    AND table_name = '{self.sensor_table.name}'
                    AND column_name = '{tag_name}'
                """
                result = connection.execute(text(check_query)).fetchone()
                if not result:
                    raise ValueError(f"Column '{tag_name}' does not exist in the database")

            q = """
                SELECT
                  MIN("{tag}") as min,
                  MAX("{tag}") as max,
                  AVG("{tag}") as avg,
                  VARIANCE("{tag}") as var
                FROM :table
            """.format(tag=tag_name)
            df = self.query(q)
        else:
            # for narrow format, check if the tag exists
            with TryConnectContextManager(self.engine) as connection:
                check_query = f"""
                    SELECT name
                    FROM {self.sensor_table.schema}.{self.sensor_table.name}
                    WHERE name = '{tag_name}'
                    LIMIT 1
                """
                result = connection.execute(text(check_query)).fetchone()
                if not result:
                    raise ValueError(f"Tag '{tag_name}' does not exist in the database")

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
