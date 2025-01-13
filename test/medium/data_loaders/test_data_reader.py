from datetime import UTC, datetime, timedelta
from typing import List

import pytest
import numpy as np
from docker.models.containers import Container
from pandas import DataFrame, Series, DatetimeIndex
import pandas as pd

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.data_pipeline.db.data_reader import TagDBConfig
from test.medium.data_loaders.test_data_writer import write_n_random_vals
from test.infrastructure.utils.docker import init_docker_container


@pytest.fixture(scope="module")
def init_data_reader_tsdb_container():
    container = init_docker_container(ports={"5432": 5433})
    yield container
    container.stop()
    container.remove()

@pytest.fixture(scope="module")
def data_reader_writer(init_data_reader_tsdb_container: Container):
    assert init_data_reader_tsdb_container.name == "test_timescale"
    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=5433,  # default is 5432, but we want to use different port for test db
        db_name="pytest",
        sensor_table_name="sensors",
    )

    data_reader = DataReader(db_cfg=db_cfg)
    data_writer = DataWriter(db_cfg=db_cfg)

    yield (data_reader, data_writer)

    data_reader.close()

class TestDataReaderLogic:
    def test_single_aggregated_read_end_time_inclusive(self, data_reader_writer: tuple[DataReader, DataWriter]):
        reader, writer = data_reader_writer
        end_time = datetime(year=2025, month=1, day=10, hour=10, minute=30, tzinfo=UTC)
        writer.write(
            timestamp=end_time, name="test_var", val=0.1
        )
        writer.blocking_sync()
        obs_period = timedelta(seconds=1)
        start_time = end_time - obs_period
        df = reader.single_aggregated_read(names=["test_var"], start_time=start_time, end_time=end_time)
        assert len(df) == 1
        assert df["test_var"].values[0] == 0.1
        assert df.index[0] == end_time

    def test_single_aggregated_read_start_time_exclusive(self, data_reader_writer: tuple[DataReader, DataWriter]):
        reader, writer = data_reader_writer
        obs_period = timedelta(seconds=2)
        end_time = datetime(year=2025, month=1, day=10, hour=11, minute=30, tzinfo=UTC)
        start_time = end_time - obs_period

        # this should be excluded
        writer.write(
            timestamp=end_time - obs_period, name="test_var", val=0.9
        )
        # this should be included
        writer.write(
            timestamp=end_time, name="test_var", val=0.1
        )
        writer.blocking_sync()
        df = reader.single_aggregated_read(names=["test_var"], start_time=start_time, end_time=end_time)
        assert len(df) == 1
        assert df["test_var"].values[0] == 0.1
        assert df.index[0] == end_time

    def test_single_aggregated_read_multi_val(self, data_reader_writer: tuple[DataReader, DataWriter]):
        reader, writer = data_reader_writer
        obs_period = timedelta(seconds=2)
        end_time = datetime(year=2025, month=1, day=10, hour=12, minute=30, tzinfo=UTC)
        start_time = end_time - obs_period
        writer.write(
            timestamp=end_time, name="test_var", val=0.1
        )
        writer.write(
            timestamp=end_time - obs_period / 2, name="test_var", val=0.9
        )
        writer.blocking_sync()
        df = reader.single_aggregated_read(names=["test_var"], start_time=start_time, end_time=end_time)
        assert len(df) == 1
        assert df["test_var"].values[0] == 0.5
        assert df.index[0] == end_time

    def test_batch_aggregated_read(self, data_reader_writer: tuple[DataReader, DataWriter]):
        # expect 5 rows returned, aggregated across 10 values in the db
        reader, writer = data_reader_writer

        obs_period = timedelta(seconds=2)
        end_time = datetime(year=2025, month=1, day=10, hour=13, minute=30, tzinfo=UTC)
        start_time = end_time - 5*obs_period # we expect 5 rows

        for i in range(10):
            write_time = end_time - i*timedelta(seconds=1)
            writer.write(
                timestamp=write_time, name="test_var", val=1.0 - i*0.1
            )
        writer.blocking_sync()

        # in chronological order: [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]
        expected_vals = [0.15, 0.35, 0.55, 0.75, 0.95]
        expected_index = DatetimeIndex(['2025-01-10 13:29:52+00:00', '2025-01-10 13:29:54+00:00',
                                        '2025-01-10 13:29:56+00:00', '2025-01-10 13:29:58+00:00',
                                        '2025-01-10 13:30:00+00:00'],
                                        dtype='datetime64[ns, UTC]', name='time_bucket')

        df = reader.batch_aggregated_read(
            names=["test_var"], start_time=start_time, end_time=end_time, bucket_width=obs_period
        )

        assert len(df) == 5
        assert np.allclose(df["test_var"].to_numpy(), np.array(expected_vals))
        assert isinstance(df.index, DatetimeIndex) # for typing
        assert (df.index.to_numpy() == expected_index.to_numpy()).all()

    def test_batch_aggregated_read_exclusive_start_time(self, data_reader_writer: tuple[DataReader, DataWriter]):
        # expect 5 rows returned, aggregated across 10 values in the db
        reader, writer = data_reader_writer

        obs_period = timedelta(seconds=2)
        end_time = datetime(year=2025, month=1, day=10, hour=14, minute=30, tzinfo=UTC)
        start_time = end_time - 5*obs_period # we expect 5 rows

        for i in range(10):
            write_time = end_time - i*timedelta(seconds=1)
            writer.write(
                timestamp=write_time, name="test_var", val=1.0 - i*0.1
            )
        # write an extra point on the start time that should be excluded
        writer.write(
            timestamp=start_time, name="test_var", val=0.0
        )
        # write an addition point before start time
        writer.write(
            timestamp=start_time - timedelta(seconds=1), name="test_var", val=-0.1
        )
        writer.blocking_sync()

        # included data should be the same as in the previous test (test_batch_aggregated_read)
        # in chronological order: [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]
        expected_vals = [0.15, 0.35, 0.55, 0.75, 0.95]
        expected_index = DatetimeIndex(['2025-01-10 14:29:52+00:00', '2025-01-10 14:29:54+00:00',
                                        '2025-01-10 14:29:56+00:00', '2025-01-10 14:29:58+00:00',
                                        '2025-01-10 14:30:00+00:00'],
                                        dtype='datetime64[ns, UTC]', name='time_bucket')

        df = reader.batch_aggregated_read(
            names=["test_var"], start_time=start_time, end_time=end_time, bucket_width=obs_period
        )

        assert len(df) == 5
        assert np.allclose(df["test_var"].to_numpy(), np.array(expected_vals))
        assert isinstance(df.index, DatetimeIndex) # for typing
        assert (df.index.to_numpy() == expected_index.to_numpy()).all()

class TestDataReader:
    sensor_names: List[str] = ["sensor1", "sensor2", "sensor3"]

    @pytest.fixture(autouse=True, scope="class")
    def now(self):
        """
        scope="class" should mean this is only executed once and the timestamp is shared
        """
        return datetime.now(UTC)

    @pytest.fixture(autouse=False, scope="class")
    def populate_db(self, data_reader_writer: tuple[DataReader, DataWriter], now: datetime):
        n_vals = 50
        names = TestDataReader.sensor_names
        _, data_writer = data_reader_writer
        for name in names:
            write_n_random_vals(n=n_vals, name=name, data_writer=data_writer, end_time=now)

    def test_read_with_no_data(self, data_reader_writer: tuple[DataReader, DataWriter], populate_db: None, now: datetime): # noqa: E501
        data_reader, _ = data_reader_writer
        end_time = now - timedelta(hours=2) # preceeds all sensor readings
        start_time = end_time - timedelta(minutes=5)

        result_df = data_reader.single_aggregated_read(
            names=TestDataReader.sensor_names, start_time=start_time, end_time=end_time
        )
        assert TestDataReader.sensor_names == result_df.columns.tolist()
        series_all_nan = result_df.isnull().all()
        assert isinstance(series_all_nan, Series)
        assert series_all_nan.all()

    def test_read_with_non_UTC(self, data_reader_writer: tuple[DataReader, DataWriter], populate_db: None):
        data_reader, _ = data_reader_writer
        end_time = datetime.now() # didn't explicitly add UTC timezone
        start_time = end_time - timedelta(minutes=5)

        result_df = data_reader.single_aggregated_read(
            names=TestDataReader.sensor_names, start_time=start_time, end_time=end_time
        )
        assert TestDataReader.sensor_names == result_df.columns.tolist()
        series_all_not_nan = result_df.notna().all()
        assert isinstance(series_all_not_nan, Series)
        assert series_all_not_nan.all()

    def test_single_avg_read(self, data_reader_writer: tuple[DataReader, DataWriter], populate_db: None, now: datetime):
        data_reader, _ = data_reader_writer
        end_time = now
        start_time = end_time - timedelta(minutes=5)
        result_df = data_reader.single_aggregated_read(
            names=TestDataReader.sensor_names, start_time=start_time, end_time=end_time, aggregation="avg"
        )
        assert TestDataReader.sensor_names == result_df.columns.tolist()

    def test_single_last_read(self, data_reader_writer: tuple[DataReader, DataWriter], populate_db: None, now: datetime): # noqa: E501
        data_reader, _ = data_reader_writer
        end_time = now
        start_time = end_time - timedelta(minutes=5)
        result_df = data_reader.single_aggregated_read(
            names=TestDataReader.sensor_names, start_time=start_time, end_time=end_time, aggregation="last"
        )
        assert TestDataReader.sensor_names == result_df.columns.tolist()

    def test_batch_avg_read(self, data_reader_writer: tuple[DataReader, DataWriter], populate_db: None, now: datetime):
        data_reader, _ = data_reader_writer
        end_time = now + timedelta(minutes=1)
        start_time = end_time - timedelta(minutes=30)
        result_df = data_reader.batch_aggregated_read(
            names=TestDataReader.sensor_names,
            start_time=start_time,
            end_time=end_time,
            bucket_width=timedelta(seconds=10),
            aggregation="avg",
        )

        self._ensure_names_included(result_df)

    def test_batch_last_read(self, data_reader_writer: tuple[DataReader, DataWriter], populate_db: None, now: datetime):
        data_reader, _ = data_reader_writer
        end_time = now + timedelta(minutes=1)
        start_time = end_time - timedelta(minutes=30)
        result_df = data_reader.batch_aggregated_read(
            names=TestDataReader.sensor_names,
            start_time=start_time,
            end_time=end_time,
            bucket_width=timedelta(seconds=10),
            aggregation="last",
        )

        self._ensure_names_included(result_df)

    def test_missing_col_batch_aggregated_read(self, data_reader_writer: tuple[DataReader, DataWriter], populate_db: None, now: datetime): # noqa: E501
        data_reader, _ = data_reader_writer
        end_time = now + timedelta(minutes=1)
        start_time = end_time - timedelta(minutes=30)
        missing_sensor_name = "sensor_x"

        result_df = data_reader.batch_aggregated_read(
            names=TestDataReader.sensor_names + [missing_sensor_name],
            start_time=start_time,
            end_time=end_time,
            bucket_width=timedelta(seconds=10),
        )

        assert bool(result_df[missing_sensor_name].isnull().all())
        self._ensure_names_included(result_df)

    def test_missing_col_single_aggregated_read(self, data_reader_writer: tuple[DataReader, DataWriter], populate_db: None, now: datetime): # noqa: E501
        data_reader, _ = data_reader_writer
        end_time = now + timedelta(minutes=1)
        start_time = end_time - timedelta(minutes=30)
        missing_sensor_name = "sensor_x"

        result_df = data_reader.single_aggregated_read(
            names=TestDataReader.sensor_names + [missing_sensor_name],
            start_time=start_time,
            end_time=end_time,
        )

        assert bool(result_df[missing_sensor_name].isnull().all())
        self._ensure_names_included(result_df)
        series_all_not_nan = result_df[TestDataReader.sensor_names].notna().all()
        assert isinstance(series_all_not_nan, Series)
        assert series_all_not_nan.all()

    def _ensure_names_included(self, data: DataFrame | Series) -> None:
        for name in TestDataReader.sensor_names:
            assert name in data.columns

    def test_batch_read_timestamp_precision(
        self,
        data_reader_writer: tuple[DataReader, DataWriter],
        populate_db: None,
        now: datetime
    ):
        data_reader, data_writer = data_reader_writer

        # timestamps with different precisions
        base_time = datetime(2024, 5, 16, 3, 15, 0, tzinfo=UTC)
        end_time = datetime(2024, 5, 16, 3, 15, 0, 123456, tzinfo=UTC)
        start_time = base_time - timedelta(minutes=4)

        # write some data but skip one timestamp to create a gap
        test_times = pd.date_range(start=start_time, end=end_time, freq=timedelta(minutes=1))
        missing_time = test_times[2]

        for name in TestDataReader.sensor_names:
            for t in test_times:
                if t != missing_time:
                    data_writer.write(name=name, val=1.0, timestamp=t)
                    data_writer.blocking_sync()

        result_df = data_reader.batch_aggregated_read(
            names=TestDataReader.sensor_names,
            start_time=start_time,
            end_time=end_time,
            bucket_width=timedelta(minutes=1),
        )

        expected_timestamps = pd.date_range(
            start=start_time + timedelta(minutes=1),
            end=end_time,
            freq=timedelta(minutes=1),
            tz='UTC'
        )

        pd.testing.assert_index_equal(result_df.index, expected_timestamps)
        assert result_df.index.is_unique
        missing_time_data = result_df.loc[missing_time]
        assert missing_time_data.isna().all()
        self._ensure_names_included(result_df)
