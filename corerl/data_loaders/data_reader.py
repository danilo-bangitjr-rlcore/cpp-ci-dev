from omegaconf import DictConfig
from datetime import datetime, timedelta, UTC
from corerl.sql_logging.sql_logging import get_sql_engine
import pandas as pd
from sqlalchemy import Engine
from typing import List
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
        assert start_time.tzinfo == UTC
        assert end_time.tzinfo == UTC
        start_t_str = start_time.isoformat()
        end_t_str = end_time.isoformat()

        name_filter = "\tOR ".join([f"name = '{name}'\n" for name in names])
        name_filter = f"({name_filter})"
        query_str = f"""
            SELECT 
              time_bucket(INTERVAL '{bucket_width}', time) as time_bucket,
              name,
              avg((fields->'val')::float) AS avg_val
            FROM {self.sensor_table_name}
            WHERE time > TIMESTAMP '{start_t_str}'
            AND time < TIMESTAMP '{end_t_str}'
            AND {name_filter}
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
        assert start_time.tzinfo == UTC
        assert end_time.tzinfo == UTC
        start_t_str = start_time.isoformat()
        end_t_str = end_time.isoformat()

        name_filter = "\tOR ".join([f"name = '{name}'\n" for name in names])
        name_filter = f"({name_filter})"
        query_str = f"""
            SELECT 
              name,
              avg((fields->'val')::float) AS avg_val
            FROM {self.sensor_table_name}
            WHERE time > TIMESTAMP '{start_t_str}'
            AND time < TIMESTAMP '{end_t_str}'
            AND {name_filter}
            GROUP BY name
            ORDER BY name ASC;
        """

        sensor_data = pd.read_sql(sql=query_str, con=self.connection)
        if sensor_data.empty:
            logger.warning(f"failed query:\n{query_str}")
            raise Exception("dataframe returned from timescale was empty.")
        sensor_data["time"] = end_time

        sensor_data = sensor_data.pivot(columns="name", values="avg_val", index="time")
        missing_cols = set(names) - set(sensor_data.columns)
        sensor_data[list(missing_cols)] = np.nan

        return sensor_data

    def close(self) -> None:
        self.connection.close()
