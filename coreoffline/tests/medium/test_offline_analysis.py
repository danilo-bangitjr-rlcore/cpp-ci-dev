import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from corerl.data_pipeline.datatypes import StageCode, Transition
from corerl.data_pipeline.pipeline import Pipeline
from corerl.state import AppState
from corerl.utils.pandas import split_dataframe_into_chunks
from lib_agent.buffer.datatypes import DataMode

from coreoffline.utils.config import OfflineMainConfig
from coreoffline.core.pipeline_utils import StageDataCapture
from coreoffline.data_analysis import data_report, transition_report
from coreoffline.data_analysis.data_report import ReportConfig


def test_stage_data_capture(
    offline_cfg: OfflineMainConfig,
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


def test_data_report_smoke_test(
        tmp_path: Path,
        dummy_app_state: AppState,
    ):
    """Smoke test for data report generation to ensure it terminates with mixed nan/non-nan data"""

    # Create test dataframes with mixed nan and non-nan values
    test_data_1 = pd.DataFrame({
        'Action': [0.0, 1.0, np.nan, 1.0, 0.0],
        'Tag_1': [1.0, np.nan, 3.0, np.nan, 5.0],
        'reward': [np.nan, 1.0, 0.0, 1.0, np.nan],
        'extra_col': [10.0, 20.0, 30.0, np.nan, 50.0],
    })

    test_data_2 = pd.DataFrame({
        'Action': [np.nan, np.nan, 0.0, 1.0, 1.0],
        'Tag_1': [7.0, 8.0, np.nan, 10.0, 11.0],
        'reward': [0.0, np.nan, np.nan, 0.0, 1.0],
        'extra_col': [np.nan, np.nan, np.nan, 40.0, 60.0],
    })

    # Create report config
    report_cfg = ReportConfig(
        output_dir=tmp_path / "data_report",
        stages=[StageCode.INIT, StageCode.SC],
        stat_table_enabled=True,
        hist_enabled=True,
        cross_corr_enabled=True,
        cross_corr_max_lag=5,  # Keep small to speed up test
        hist_num_bins=10,  # Keep small to speed up test
        contiguous_time_threshold=dt.timedelta(minutes=1),
    )

    data_list = [test_data_1, test_data_2]
    stages = [StageCode.INIT, StageCode.SC]
    start = dt.datetime.now()
    end = start + dt.timedelta(minutes=6)

    # This should not raise any exceptions - we don't care about the output
    data_report.generate_report(report_cfg, data_list, stages, dummy_app_state, start, end)

    # Verify that output directory was created (basic sanity check)
    assert (tmp_path / "data_report").exists()

    # Verify that sensor report was created
    assert (tmp_path / "data_report" / "sensor_report.txt").exists()


def test_transition_report_smoke_test(
        tmp_path: Path,
        transitions_with_timestamps: list[Transition],
        dummy_app_state: AppState,
    ):
    """Smoke test for transition report generation to ensure it terminates correctly"""

    # Create report config
    report_cfg = ReportConfig(
        output_dir=tmp_path / "transition_report",
        stages=[StageCode.INIT, StageCode.SC],
        stat_table_enabled=False,
        hist_enabled=False,
        cross_corr_enabled=False,
        contiguous_time_threshold=dt.timedelta(minutes=1),
    )

    start = dt.datetime(2023, 7, 13, 9, 59, tzinfo=dt.UTC)
    end = dt.datetime(2023, 7, 13, 10, 5, tzinfo=dt.UTC)

    # This should not raise any exceptions
    transition_report.generate_report(
        report_cfg,
        dummy_app_state,
        start,
        end,
        transitions_with_timestamps,
    )

    # Verify that output directory was created
    assert (tmp_path / "transition_report").exists()

    # Verify that transition statistics file was created
    assert (tmp_path / "transition_report" / "transition_statistics.txt").exists()
    assert (tmp_path / "transition_report" / "transition_statistics.csv").exists()
