from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import Engine, text

from corerl.data_pipeline.db.data_reader import Agg, DataReader
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig


@pytest.fixture()
def wide_format_db(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    port = tsdb_engine.url.port
    assert port is not None

    with tsdb_engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS public.wide_sensors CASCADE"))
        conn.commit()

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

        # Define tag_aggregations for each sensor
        tag_aggregations = {
            "sensor_a": Agg.avg,
            "sensor_b": Agg.avg,
            "sensor_c": Agg.bool_or
        }

        result_df = reader.batch_aggregated_read(
            names=["sensor_a", "sensor_b", "sensor_c"],
            start_time=min_time - timedelta(seconds=1),
            end_time=max_time + timedelta(seconds=1),
            bucket_width=timedelta(seconds=10),
            tag_aggregations=tag_aggregations
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
            end_time=base_time,
            tag_aggregations={"sensor_a": Agg.avg}
        )

        assert single_df["sensor_a"].iloc[0] == 99.9

    def test_edge_cases(self, wide_format_db: tuple[DataWriter, DataReader]):
        writer, reader = wide_format_db
        base_time = datetime.now(UTC).replace(microsecond=0)

        # writing with different primitive types
        writer.write(base_time, "float_sensor", 3.14)
        writer.write(base_time, "int_sensor", 42)
        writer.write(base_time, "bool_sensor", True)
        writer.write(base_time, "text_sensor", "word")

        writer.write(base_time, "new_column", 123.45)
        writer.write(base_time, "null_sensor", None)

        writer.flush()

        # reading with different aggregations
        tag_aggregations = {
            "bool_sensor": Agg.bool_or,
            "int_sensor": Agg.avg,
            "float_sensor": Agg.avg,
            "text_sensor": Agg.last,
            "new_column": Agg.avg,
            "null_sensor": Agg.avg,
        }

        result_df = reader.batch_aggregated_read(
            names=list(tag_aggregations.keys()),
            start_time=base_time - timedelta(seconds=5),
            end_time=base_time + timedelta(seconds=5),
            bucket_width=timedelta(seconds=10),
            tag_aggregations=tag_aggregations
        )

        assert not result_df.empty, "Result dataframe should not be empty"
        assert result_df["bool_sensor"].iloc[0]
        assert result_df["int_sensor"].iloc[0] == 42
        assert result_df["float_sensor"].iloc[0] == 3.14
        assert pd.isna(result_df["null_sensor"].iloc[0])

        # reading a column that doesn't exist in DB
        with pytest.raises(ValueError):
            reader.batch_aggregated_read(
                names=["non_existent_column"],
                start_time=base_time - timedelta(seconds=1),
                end_time=base_time + timedelta(seconds=1),
                bucket_width=timedelta(seconds=10),
                tag_aggregations={"non_existent_column": Agg.avg}
            )

        # column with some nulls
        writer.write(base_time - timedelta(seconds=10), "partial_null", 1.0)
        writer.write(base_time, "partial_null", None)
        writer.flush()

        partial_null_df = reader.batch_aggregated_read(
            names=["partial_null"],
            start_time=base_time - timedelta(seconds=20),
            end_time=base_time + timedelta(seconds=1),
            bucket_width=timedelta(seconds=10),
            tag_aggregations={"partial_null": Agg.avg}
        )

        assert len(partial_null_df) > 1
        assert not pd.isna(partial_null_df["partial_null"].iloc[0])
        assert pd.isna(partial_null_df["partial_null"].iloc[1])

        # get_tag_stats function
        stats = reader.get_tag_stats("float_sensor")
        assert stats.min == 3.14
        assert stats.max == 3.14
        assert stats.avg == 3.14

        # get_tag_stats on non-existent column
        with pytest.raises(ValueError):
            reader.get_tag_stats("non_existent_column")

        # get_tag_stats on column with all nulls
        all_null_stats = reader.get_tag_stats("null_sensor")
        assert pd.isna(all_null_stats.min)
        assert pd.isna(all_null_stats.max)
        assert pd.isna(all_null_stats.avg)
