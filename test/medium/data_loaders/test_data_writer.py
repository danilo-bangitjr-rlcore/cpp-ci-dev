from datetime import UTC, datetime, timedelta
from random import random
from typing import Generator

import pytest
from sqlalchemy import Engine

import corerl.utils.nullable as nullable
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.data_pipeline.db.data_writer import DataWriter


@pytest.fixture()
def data_writer(tsdb_engine: Engine, tsdb_tmp_db_name: str) -> Generator[DataWriter, None, None]:
    assert tsdb_engine.url.port is not None
    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=tsdb_engine.url.port,
        db_name=tsdb_tmp_db_name,
        table_name="sensors",
    )

    data_writer = DataWriter(cfg=db_cfg)

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
        data_writer.blocking_sync()


def test_writing_datapt(data_writer: DataWriter):
    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)


def test_batch_write(data_writer: DataWriter):
    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    for _ in range(10):
        sensor_val += 1
        data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)


def test_microsecond_trimming(data_writer: DataWriter):
    ts = datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)
    data_writer.write(timestamp=ts, name="test_sensor", val=1.0)
    assert data_writer._buffer[-1].ts == "2024-01-01T12:00:00+00:00"
