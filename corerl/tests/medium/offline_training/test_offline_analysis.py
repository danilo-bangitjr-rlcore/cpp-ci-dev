import datetime as dt

import pandas as pd

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.offline_analysis import StageDataCapture
from corerl.state import AppState
from corerl.utils.pandas import split_dataframe_into_chunks


def test_stage_data_capture(
    offline_cfg: MainConfig,
    dummy_app_state: AppState,
):
    """Test that StageDataCapture works with pipeline execution and that INIT stage data equals original"""

    # Create test dataframe similar to what load_entire_dataset would return
    test_dataframe = pd.DataFrame({
        'Action': [0, 1, 0, 1, 0] * 4,  # 20 rows total
        'Tag_1': [1, 2, 3, 4, 5] * 4,
        'reward': [0, 1, 0, 1, 0] * 4,
    }, index=pd.DatetimeIndex([
        dt.datetime(2023, 7, 13, 10, i, tzinfo=dt.UTC) for i in range(20)
    ]))

    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)
    capture = StageDataCapture(pipeline)

    data_chunks = split_dataframe_into_chunks(test_dataframe, 10)

    # Run pipeline on chunks
    for chunk in data_chunks:
        pipeline(
            data=chunk,
            data_mode=DataMode.OFFLINE,
            reset_temporal_state=False,
        )

    # Verify data was captured for each stage
    for data in capture.captured_data.values():
        assert len(data) > 0

    init_captured = capture.get_concatenated_data(StageCode.INIT)

    # there are some type casts in the pipeline, so don't check types
    pd.testing.assert_frame_equal(init_captured, test_dataframe, check_dtype=False)

