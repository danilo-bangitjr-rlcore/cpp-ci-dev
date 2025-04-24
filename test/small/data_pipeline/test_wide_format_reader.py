from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import Engine, text

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig


@pytest.fixture()
def wide_format_db(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    port = tsdb_engine.url.port
    assert port is not None

    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=port,
        db_name=tsdb_tmp_db_name,
        table_name='wide_sensors',
        table_schema='public',
        wide_format=True
    )

    writer = DataWriter(db_cfg)
    reader = DataReader(db_cfg)

    yield writer, reader

    with tsdb_engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS public.wide_sensors CASCADE"))
        conn.commit()


class TestWideFormatWriter:
    def test_write_and_read(self, wide_format_db: tuple[DataWriter, DataReader]):
        writer, reader = wide_format_db
        base_time = datetime.now(UTC).replace(microsecond=0)
        timestamps = []
        for i in range(10):
            timestamp = base_time - timedelta(seconds=i*10)
            timestamps.append(timestamp)
            writer.write(timestamp, "sensor_a", 10.0 + i)
            writer.write(timestamp, "sensor_b", 20.0 + i)
            writer.write(timestamp, "sensor_c", i % 2 == 0)

        writer.flush()

        min_time = min(timestamps)
        max_time = max(timestamps)

        result_df = reader.batch_aggregated_read(
            names=["sensor_a", "sensor_b", "sensor_c"],
            start_time=min_time - timedelta(seconds=1),
            end_time=max_time + timedelta(seconds=1),
            bucket_width=timedelta(seconds=10)
        )


        assert not result_df.empty, "Result dataframe should not be empty"
        assert "sensor_a" in result_df.columns
        assert "sensor_b" in result_df.columns
        assert "sensor_c" in result_df.columns

        assert pd.api.types.is_numeric_dtype(result_df["sensor_a"])
        assert pd.api.types.is_numeric_dtype(result_df["sensor_b"])
        assert pd.api.types.is_bool_dtype(result_df["sensor_c"])

        writer.write(base_time, "sensor_a", 99.9)
        writer.flush()

        single_df = reader.single_aggregated_read(
            names=["sensor_a"],
            start_time=base_time - timedelta(seconds=1),
            end_time=base_time
        )

        assert single_df["sensor_a"].iloc[0] == 99.9
