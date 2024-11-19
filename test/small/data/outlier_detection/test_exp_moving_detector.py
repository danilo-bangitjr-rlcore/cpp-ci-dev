from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from corerl.data.data import PipelineFrame
from corerl.data_pipeline.outlier_detectors.exp_moving_detector import ExpMovingDetector


def test_obvious_outlier_in_first_batch():
    """
    The initial variance is calculated using the first batch,
    so we need a large batch in this test in order to ensure
    the outlier doesn't significantly poison the variance.

    If the online stats are initialized with outlier-free data,
    the outlier can be detected in a smaller subsequent batch
    """
    outlier_detector = ExpMovingDetector(alpha=0.99)
    values = [1] * 1000
    values[-1] = 100  # <- this is the outlier
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data)
    filtered_pf = outlier_detector.filter(pf)
    filtered_data = filtered_pf.data

    assert not np.isnan(filtered_data[name].iloc[-2])
    assert np.isnan(filtered_data[name].iloc[-1])


def test_obvious_outlier_in_second_batch():
    """
    Note the batches here are smaller compared to the previous
    test (test_obvious_outlier_in_first_batch)
    """
    outlier_detector = ExpMovingDetector(alpha=0.99)
    values = [1] * 5
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data)
    outlier_detector.filter(pf)  # <- stats get initialized here

    values2 = [1] * 5
    values2[-1] = 100  # <- this is the outlier

    data2 = pd.DataFrame({name: values2})
    pf2 = PipelineFrame(data2)
    filtered_pf2 = outlier_detector.filter(pf2)
    filtered_data2 = filtered_pf2.data

    assert not np.isnan(filtered_data2[name].iloc[-2])
    assert np.isnan(filtered_data2[name].iloc[-1])


def test_obvious_outlier_in_stream():
    outlier_detector = ExpMovingDetector(alpha=0.99)
    name = "sensor_x"

    for _ in range(10):
        values = [1]
        data = pd.DataFrame({name: values})
        pf = PipelineFrame(data)

        filtered_pf = outlier_detector.filter(pf)
        filtered_data = filtered_pf.data

        assert not np.isnan(filtered_data[name].iloc[0])

    # catch the outlier
    values = [10]  # <- this is the outlier
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data)

    filtered_pf = outlier_detector.filter(pf)
    filtered_data = filtered_pf.data

    assert np.isnan(filtered_data[name].iloc[0])


def test_detection_with_multiple_cols():
    outlier_detector = ExpMovingDetector(alpha=0.99)
    name_x = "sensor_x"
    name_y = "sensor_y"

    for _ in range(10):
        values_x = [1]
        values_y = [2]

        data = pd.DataFrame({name_x: values_x, name_y: values_y})
        pf = PipelineFrame(data)

        filtered_pf = outlier_detector.filter(pf)
        filtered_data = filtered_pf.data
        assert not filtered_data.isnull().values.any()

    # catch the outlier
    values_x = [10]  # <- this is the outlier
    values_y = [2]  # <- this is not an outlier

    data = pd.DataFrame({name_x: values_x, name_y: values_y})
    pf = PipelineFrame(data)

    filtered_pf = outlier_detector.filter(pf)
    filtered_data = filtered_pf.data

    assert np.isnan(filtered_data[name_x].iloc[0])
    assert not np.isnan(filtered_data[name_y].iloc[0])


def test_detector_does_not_change_indices():
    outlier_detector = ExpMovingDetector(alpha=0.99)
    n = 5
    values = [1] * n
    base_timestamp = datetime.strptime("31/01/24 23:59:59", "%d/%m/%y %H:%M:%S")
    timestamps = [base_timestamp + i * timedelta(minutes=5) for i in range(n)]
    name = "sensor_x"

    data = pd.DataFrame({name: values}, index=pd.DatetimeIndex(timestamps))
    pf = PipelineFrame(data)

    filtered_pf = outlier_detector.filter(pf)
    filtered_data = filtered_pf.data

    for i, dt_index in enumerate(filtered_data.index):
        ts = pd.to_datetime(dt_index)
        assert ts == timestamps[i]

    # catch the outlier
    values = [10]  # <- this is the outlier
    outlier_ts = timestamps[-1] + timedelta(minutes=5)
    data = pd.DataFrame({name: values}, index=pd.DatetimeIndex([outlier_ts]))
    pf = PipelineFrame(data)

    filtered_pf = outlier_detector.filter(pf)
    filtered_data = filtered_pf.data

    assert np.isnan(filtered_data[name].iloc[0])

    ts = pd.to_datetime(filtered_data.index[0])
    assert ts == outlier_ts
