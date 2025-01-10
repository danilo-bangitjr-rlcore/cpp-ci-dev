from docker.models.containers import Container
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.data_pipeline.db.data_reader import TagDBConfig
from datetime import datetime, UTC, timedelta
import pytest
from typing import Generator
from random import random

import corerl.utils.nullable as nullable
from test.infrastructure.utils.docker import init_docker_container


@pytest.fixture(scope="module")
def init_data_writer_tsdb_container():
    container = init_docker_container(ports={"5432": 5433})
    yield container
    container.stop()
    container.remove()


@pytest.fixture(scope="module")
def data_writer(init_data_writer_tsdb_container: Container) -> Generator[DataWriter, None, None]:
    assert init_data_writer_tsdb_container.name == "test_timescale"
    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=5433, # default is 5432, but we want to use different port for test db
        db_name="pytest",
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
