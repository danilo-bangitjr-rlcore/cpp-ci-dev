from datetime import UTC, datetime, timedelta
from random import random

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.utils import nullable


def write_n_random_vals(
    n: int,
    name: str,
    data_reader_writer: tuple[DataReader, DataWriter],
    end_time: datetime | None = None,
    interval: timedelta = timedelta(seconds=5),
):
    _, data_writer = data_reader_writer
    end_time = nullable.default(end_time, lambda: datetime.now(UTC))
    for i in range(n, 0, -1):
        ts = end_time - i * interval
        val = random()
        data_writer.write(timestamp=ts, name=name, val=val)
        data_writer.blocking_sync()


def test_writing_datapt(data_reader_writer: tuple[DataReader, DataWriter]):
    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    _, data_writer = data_reader_writer
    data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)


def test_batch_write(data_reader_writer: tuple[DataReader, DataWriter]):
    _, data_writer = data_reader_writer

    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    for _ in range(10):
        sensor_val += 1
        data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)
