from corerl.data_loaders.data_writer import DataWriter
from omegaconf import OmegaConf
from datetime import datetime, UTC, timedelta
import pytest
from typing import Generator
from random import random

import corerl.utils.nullable as nullable


@pytest.fixture(scope="module")
def data_writer() -> Generator[DataWriter, None, None]:
    db_cfg = OmegaConf.create(
        {
            "drivername": "postgresql+psycopg2",
            "username": "postgres",
            "password": "password",
            "ip": "localhost",
            "port": 5433,  # default is 5432, but we want to use different port for test db
        }
    )

    db_name = "pytest"
    sensor_table_name = "sensors"
    data_writer = DataWriter(db_cfg=db_cfg, db_name=db_name, sensor_table_name=sensor_table_name, commit_every=1)

    yield data_writer

    data_writer.close()


def write_n_random_vals(
    n: int,
    name: str,
    data_writer: DataWriter,
    end_time: datetime | None = None,
    interval: timedelta = timedelta(seconds=5),
):
    end_time = nullable.default(end_time, lambda: datetime.now(UTC))
    for i in range(n, 0, -1):
        ts = end_time - i * interval
        val = random()
        data_writer.write(timestamp=ts, name=name, val=val)


@pytest.mark.skip(reason="github actions do not yet support docker")
def test_writing_datapt(data_writer: DataWriter):
    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)
    data_writer.commit()


@pytest.mark.skip(reason="github actions do not yet support docker")
def test_batch_write(data_writer: DataWriter):
    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    data_writer.commit_every = 10

    for _ in range(10):
        sensor_val += 1
        data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)
