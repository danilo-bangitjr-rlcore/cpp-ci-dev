import datetime
from pathlib import Path

import pandas as pd
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import TagDBConfig

from coreoffline.examples.ingest_csv import load_dataset
from coreoffline.utils.config import LoadDataConfig


def test_ingest_csv_smoke(test_db_config: TagDBConfig, tmp_path: Path):
    """
    Smoke test: Verify basic CSV ingestion works end-to-end.
    """
    csv_path = tmp_path / "test_data.csv"
    test_data = pd.DataFrame({
        "Date": [
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
            "2024-01-01 03:00:00",
        ],
        "sensor_1": [1.0, 2.0, 3.0],
        "action_1": [4.0, 5.0, 6.0],
    })
    test_data.to_csv(csv_path, index=False)

    # Create config for ingestion
    cfg = LoadDataConfig(
        csv_path=csv_path,
        reward_tags=[],
        action_tags=["action_1"],
        input_tags=["sensor_1"],
        data_writer=test_db_config,
    )

    # Ingest the data
    load_dataset(cfg)

    # Verify data was written correctly
    reader = DataReader(test_db_config)

    # Read data back from database
    start = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)
    end = datetime.datetime(2024, 1, 1, 3, 0, 0, tzinfo=datetime.UTC)

    data_df = reader.batch_aggregated_read(
        ["sensor_1", "action_1"],
        start,
        end,
        bucket_width=datetime.timedelta(hours=1),
    )

    # Check if the columns have the expected values
    assert len(data_df) == 3, f"Expected 3 rows, got {len(data_df)}"
    assert list(data_df["sensor_1"]) == [1.0, 2.0, 3.0], f"sensor_1 values don't match: {list(data_df['sensor_1'])}"
    assert list(data_df["action_1"]) == [4.0, 5.0, 6.0], f"action_1 values don't match: {list(data_df['action_1'])}"
