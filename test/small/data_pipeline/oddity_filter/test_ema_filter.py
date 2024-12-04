from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import MissingType
from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode
from corerl.data_pipeline.oddity_filters.ema_filter import EMAFilter, EMAFilterConfig


def test_obvious_outlier_in_first_batch():
    """
    The initial variance is calculated using the first batch,
    so we need a large batch in this test in order to ensure
    the outlier doesn't significantly poison the variance.

    If the online stats are initialized with outlier-free data,
    the outlier can be detected in a smaller subsequent batch
    """

    cfg = EMAFilterConfig(alpha=0.99)
    outlier_detector = EMAFilter(cfg)
    values = [1.0] * 1000
    values[-1] = 100.0  # <- this is the outlier
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, CallerCode.ONLINE)
    filtered_pf = outlier_detector(pf, name)
    filtered_data = filtered_pf.data

    assert not np.isnan(filtered_data[name].iloc[-2])
    assert np.isnan(filtered_data[name].iloc[-1])


def test_obvious_outlier_in_second_batch():
    """
    Note the batches here are smaller compared to the previous
    test (test_obvious_outlier_in_first_batch)
    """
    cfg = EMAFilterConfig(alpha=0.99)

    outlier_detector = EMAFilter(cfg)
    values = [1.0] * 5
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, CallerCode.ONLINE)
    pf = outlier_detector(pf, name)  # <- stats get initialized here

    values2 = [1.0] * 5
    values2[-1] = 100.0  # <- this is the outlier

    data2 = pd.DataFrame({name: values2})
    pf2 = PipelineFrame(data=data2, caller_code=CallerCode.ONLINE, temporal_state=pf.temporal_state)
    filtered_pf2 = outlier_detector(pf2, name)
    filtered_data2 = filtered_pf2.data

    assert not np.isnan(filtered_data2[name].iloc[-2])
    assert np.isnan(filtered_data2[name].iloc[-1])


def test_obvious_outlier_in_stream():
    cfg = EMAFilterConfig(alpha=0.99)

    outlier_detector = EMAFilter(cfg)
    name = "sensor_x"

    values = [1.0]
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, CallerCode.ONLINE)

    for _ in range(10):
        pf = outlier_detector(pf, name)
        filtered_data = pf.data
        assert not np.isnan(filtered_data[name].iloc[0])

    # catch the outlier
    values = [10.0]  # <- this is the outlier
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data=data, caller_code=CallerCode.ONLINE, temporal_state=pf.temporal_state)

    filtered_pf = outlier_detector(pf, name)
    filtered_data = filtered_pf.data

    assert np.isnan(filtered_data[name].iloc[0])


def test_detection_with_multiple_cols():
    cfg = EMAFilterConfig(alpha=0.99)

    outlier_detector_x = EMAFilter(cfg)
    outlier_detector_y = EMAFilter(cfg)
    name_x = "sensor_x"
    name_y = "sensor_y"

    values_x = [1.0]
    values_y = [2.0]
    data = pd.DataFrame({name_x: values_x, name_y: values_y})
    pf = PipelineFrame(data, CallerCode.ONLINE)

    for _ in range(10):
        for tag, detector in zip([name_x, name_y], [outlier_detector_x, outlier_detector_y], strict=True):
            pf = detector(pf, tag)

        filtered_data = pf.data
        assert not filtered_data.isnull().values.any()

    # catch the outlier
    values_x = [10.0]  # <- this is the outlier
    values_y = [2.0]  # <- this is not an outlier

    data = pd.DataFrame({name_x: values_x, name_y: values_y})
    pf = PipelineFrame(data=data, caller_code=CallerCode.ONLINE, temporal_state=pf.temporal_state)

    for tag, detector in zip([name_x, name_y], [outlier_detector_x, outlier_detector_y], strict=True):
        pf = detector(pf, tag)
    filtered_data = pf.data

    assert np.isnan(filtered_data[name_x].iloc[0])
    assert not np.isnan(filtered_data[name_y].iloc[0])


def test_detector_does_not_change_indices():
    cfg = EMAFilterConfig(alpha=0.99)

    outlier_detector = EMAFilter(cfg)
    n = 5
    values = [1.0] * n
    base_timestamp = datetime.strptime("31/01/24 23:59:59", "%d/%m/%y %H:%M:%S")
    timestamps = [base_timestamp + i * timedelta(minutes=5) for i in range(n)]
    name = "sensor_x"

    data = pd.DataFrame({name: values}, index=pd.DatetimeIndex(timestamps))
    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = outlier_detector(pf, name)
    filtered_data = pf.data

    for i, dt_index in enumerate(filtered_data.index):
        ts = pd.to_datetime(dt_index)
        assert ts == timestamps[i]

    # catch the outlier
    values = [10.0]  # <- this is the outlier
    outlier_ts = timestamps[-1] + timedelta(minutes=5)
    data = pd.DataFrame({name: values}, index=pd.DatetimeIndex([outlier_ts]))
    pf = PipelineFrame(data=data, caller_code=CallerCode.ONLINE, temporal_state=pf.temporal_state)

    filtered_pf = outlier_detector(pf, name)
    filtered_data = filtered_pf.data

    assert np.isnan(filtered_data[name].iloc[0])

    ts = pd.to_datetime(filtered_data.index[0])
    assert ts == outlier_ts


def test_outlier_gets_correct_missingtype():
    # initialize outlier detector
    cfg = EMAFilterConfig(alpha=0.99)

    # prepare some data to warm up the stats
    outlier_detector = EMAFilter(cfg)
    values = [1.0] * 5
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, CallerCode.ONLINE)
    outlier_detector(pf, name)  # <- stats get initialized here

    # create a batch with an outlier
    values2 = [1.0] * 5
    values2[-1] = 100.0  # <- this is the outlier

    data = pd.DataFrame({name: values2})
    pf = PipelineFrame(data=data, caller_code=CallerCode.ONLINE, temporal_state=pf.temporal_state)

    # filter the outlier
    filtered_pf = outlier_detector(pf, name)

    # check that the outlier has the correct missing type
    missing_info = filtered_pf.missing_info
    assert missing_info["sensor_x"].iloc[-1] == MissingType.OUTLIER


def test_outlier_missing_type_is_added_to_existing_missing():
    # initialize outlier detector
    cfg = EMAFilterConfig(alpha=0.99)

    # prepare some data to warm up the stats
    outlier_detector = EMAFilter(cfg)
    values = [1.0] * 5
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, CallerCode.ONLINE)
    outlier_detector(pf, name)  # <- stats get initialized here

    # create a batch with an outlier
    values2 = [1.0] * 5
    values2[-1] = 100.0  # <- this is the outlier

    data = pd.DataFrame({name: values2})
    pf = PipelineFrame(data=data, caller_code=CallerCode.ONLINE, temporal_state=pf.temporal_state)

    # add an initial missing type to the outlier
    pf.missing_info.loc[4, "sensor_x"] = MissingType.BOUNDS

    # filter the outlier
    filtered_pf = outlier_detector(pf, name)

    # check that the outlier has both missing types
    missing_info = filtered_pf.missing_info
    bitmap = MissingType(missing_info["sensor_x"].iloc[-1])

    assert bitmap == MissingType.BOUNDS | MissingType.OUTLIER
    assert MissingType.BOUNDS in bitmap
    assert MissingType.OUTLIER in bitmap
