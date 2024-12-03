from datetime import UTC, datetime, timedelta
from typing import Generator, List

import pytest
from pandas import DataFrame, Series

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.sql_logging.sql_logging import SQLEngineConfig
from test.medium.data_loaders.test_data_writer import data_writer, write_n_random_vals  # noqa: F401
from test.medium.data_loaders.utils import timescale_docker  # noqa: F401


@pytest.fixture(scope="module")
def data_reader(timescale_docker) -> Generator[DataReader, None, None]:  # noqa: F811
    db_cfg = SQLEngineConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=5433,  # default is 5432, but we want to use different port for test db
    )

    db_name = "pytest"
    sensor_table_name = "sensors"
    data_reader = DataReader(db_cfg=db_cfg, db_name=db_name, sensor_table_name=sensor_table_name)

    yield data_reader

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
    def populate_db(self, data_writer: DataWriter, now):  # noqa: F811
        n_vals = 50
        names = TestDataReader.sensor_names
        for name in names:
            write_n_random_vals(n=n_vals, name=name, data_writer=data_writer, end_time=now)

    @pytest.mark.skip(reason="github actions do not yet support docker")
    def test_read_with_non_UTC(self, data_reader: DataReader, populate_db):
        end_time = datetime.now()  # didn't explicitly add UTC timezone
        start_time = end_time - timedelta(minutes=5)

        with pytest.raises(AssertionError):
            data_reader.single_aggregated_read(
                names=TestDataReader.sensor_names, start_time=start_time, end_time=end_time
            )

    @pytest.mark.skip(reason="github actions do not yet support docker")
    def test_single_avg_read(self, data_reader: DataReader, populate_db, now):
        end_time = now
        start_time = end_time - timedelta(minutes=5)
        result_df = data_reader.single_aggregated_read(
            names=TestDataReader.sensor_names, start_time=start_time, end_time=end_time, aggregation="avg"
        )
        assert TestDataReader.sensor_names == result_df.columns.tolist()

    @pytest.mark.skip(reason="github actions do not yet support docker")
    def test_single_last_read(self, data_reader: DataReader, populate_db, now):
        end_time = now
        start_time = end_time - timedelta(minutes=5)
        result_df = data_reader.single_aggregated_read(
            names=TestDataReader.sensor_names, start_time=start_time, end_time=end_time, aggregation="last"
        )
        assert TestDataReader.sensor_names == result_df.columns.tolist()

    @pytest.mark.skip(reason="github actions do not yet support docker")
    def test_batch_avg_read(self, data_reader: DataReader, populate_db, now):
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

    @pytest.mark.skip(reason="github actions do not yet support docker")
    def test_batch_last_read(self, data_reader: DataReader, populate_db, now):
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

    @pytest.mark.skip(reason="github actions do not yet support docker")
    def test_missing_col_batch_aggregated_read(self, data_reader: DataReader, populate_db, now):
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

    @pytest.mark.skip(reason="github actions do not yet support docker")
    def test_missing_col_single_aggregated_read(self, data_reader: DataReader, populate_db, now):
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

    def _ensure_names_included(self, data: DataFrame | Series) -> None:
        for name in TestDataReader.sensor_names:
            assert name in data.columns
