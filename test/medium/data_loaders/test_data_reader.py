from datetime import UTC, datetime, timedelta
from typing import List

import pytest
from docker.models.containers import Container
from pandas import DataFrame, Series

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.data_pipeline.db.data_reader import TagDBConfig
from test.medium.data_loaders.test_data_writer import write_n_random_vals
from test.infrastructure.utils.docker import init_docker_container


@pytest.fixture(scope="class")
def init_data_reader_tsdb_container():
    container = init_docker_container(ports={"5432": 5433})
    yield container
    container.stop()
    container.remove()

@pytest.fixture(scope="class")
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
