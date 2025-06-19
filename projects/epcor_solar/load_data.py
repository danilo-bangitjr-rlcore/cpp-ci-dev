import datetime as dt
from dataclasses import dataclass
from typing import Any

import utils.data as utils

from corerl.config import DBConfig
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig

@dataclass
class SolarDataLoaderConfig:
    db_cfg: TagDBConfig
    obs_period: dt.timedelta
    bess1: bool
    bess2: bool
    std_init: float
    std_mult: float


def _load_dataset_from_s3(cfg: SolarDataLoaderConfig) -> list[tuple[Any, ...]]:
    """
    Read csv files from s3, preprocess the data, and convert into (timestamp, tag, value) tuples
    """
    offline_files = utils.get_s3_files()
    sql_tups = utils.load_csv_files(cfg, offline_files)

    return sql_tups

def load_dataset(cfg: SolarDataLoaderConfig):
    """
    Load historical data from s3, preprocess data, and write (timestamp, tag_name, value) tuples to sensors db
    """
    sql_tups = _load_dataset_from_s3(cfg)

    writer = DataWriter(
        cfg.db_cfg,
        low_watermark=1_000,
        high_watermark=10_000,
    )

    for sql_tup in sql_tups:
        writer.write(timestamp=sql_tup[0], name=sql_tup[2], val=float(sql_tup[1]))

    writer.close()

if __name__ == '__main__':
    dataloader_cfg = SolarDataLoaderConfig(
        db_cfg=TagDBConfig(
            drivername=DBConfig.drivername,
            username=DBConfig.username,
            password=DBConfig.password,
            ip=DBConfig.ip,
            port=DBConfig.port,
            db_name='epcor_solar',
            table_schema=DBConfig.schema,
            wide_format=False,
        ),
        obs_period=dt.timedelta(minutes=5),
        bess1=True,
        bess2=True,
        std_init=0.01,
        std_mult=1.05,
    )

    load_dataset(dataloader_cfg)
