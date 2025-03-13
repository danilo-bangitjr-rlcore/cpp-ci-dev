from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import DataMode, MissingType, PipelineFrame
from corerl.data_pipeline.oddity_filters.ema_filter import EMAFilter, EMAFilterConfig
from corerl.state import AppState


def test_filter_no_warmup(dummy_app_state: AppState):
    name = "sensor_x"

    # filter with no warmup
    cfg = EMAFilterConfig(alpha=0.99, warmup=0)
    outlier_detector = EMAFilter(cfg, dummy_app_state)

    values = np.array([1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 10.0])
    expected = np.array([1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0, np.nan])  # 5 and 10 are outliers
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)

    pf = outlier_detector(pf, name)
    filtered_data = pf.data[name].to_numpy()
    assert np.allclose(filtered_data, expected, equal_nan=True)

def test_filter_warmup(dummy_app_state: AppState):
    name = "sensor_x"

    # filter with warmup = 5
    cfg = EMAFilterConfig(alpha=0.99, warmup=5)
    outlier_detector = EMAFilter(cfg, dummy_app_state)

    values_0 = np.array([1.0, 1.0, 1.0])
    expected_0 = np.array([1.0, 1.0, 1.0])

    data_0 = pd.DataFrame({name: values_0})
    pf_0 = PipelineFrame(data_0, DataMode.ONLINE)
    pf_0_out = outlier_detector(pf_0, name)
    filtered_data_0 = pf_0_out.data[name].to_numpy()

    assert np.allclose(filtered_data_0, expected_0, equal_nan=True)

    warmup_vals = np.array([1.0, 1.0, 1.0, 5.0, 1.0]) # first 5 values including 2 from values_1
    expected_warmup_mean = warmup_vals.mean()
    expected_warmup_std = np.sqrt(warmup_vals.var())
    epsilon = 0.01
    outlier = expected_warmup_mean + outlier_detector.tolerance * expected_warmup_std + epsilon

    # 5 should not be removed because warmup has not finished
    values_1 = np.array([5.0, 1.0, outlier, 1.0, 1.0])
    expected_1 = np.array([5.0, 1.0, np.nan, 1.0, 1.0])

    data_1 = pd.DataFrame({name: values_1})
    pf_1 = PipelineFrame(data_1, DataMode.ONLINE, temporal_state=pf_0_out.temporal_state)
    pf_1_out = outlier_detector(pf_1, name)
    filtered_data_1 = pf_1_out.data[name].to_numpy()

    assert np.allclose(filtered_data_1, expected_1, equal_nan=True)

def test_leading_nan_data(dummy_app_state: AppState):
    cfg = EMAFilterConfig(alpha=0.99, warmup=5)
    outlier_detector = EMAFilter(cfg, dummy_app_state)

    name = "sensor_x"
    values = np.array([np.nan] + [1.0] * 10 + [5.0])  # 5 at the end is outlier
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)

    pf = outlier_detector(pf, name)
    filtered_data = pf.data

    assert np.isnan(filtered_data[name].iloc[0])  # first val remains nan
    assert not np.isnan(filtered_data[name].iloc[1:10]).all()
    assert np.isnan(filtered_data[name].iloc[-1])  # filter outlier

    missing_info = pf.missing_info
    # should not mark existing nan as oddity
    assert missing_info["sensor_x"].iloc[0] == MissingType.NULL
    assert missing_info["sensor_x"].iloc[-1] == MissingType.OUTLIER


def test_trailing_nan_data(dummy_app_state: AppState):
    cfg = EMAFilterConfig(alpha=0.99)
    outlier_detector = EMAFilter(cfg, dummy_app_state)

    name = "sensor_x"
    values = np.array([1.0] * 10 + [5.0] + [np.nan])  # 5 at the end is outlier
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)

    pf = outlier_detector(pf, name)
    filtered_data = pf.data

    assert not np.isnan(filtered_data[name].iloc[:10]).all()
    assert np.isnan(filtered_data[name].iloc[-2])  # filter outlier
    assert np.isnan(filtered_data[name].iloc[-1])  # trailing nan remains nan

    missing_info = pf.missing_info
    # should not mark existing nan as oddity
    assert missing_info["sensor_x"].iloc[-1] == MissingType.NULL
    assert missing_info["sensor_x"].iloc[-2] == MissingType.OUTLIER


def test_full_nan_data(dummy_app_state: AppState):
    cfg = EMAFilterConfig(alpha=0.99)
    outlier_detector = EMAFilter(cfg, dummy_app_state)
    name = "sensor_x"

    values = np.array([np.nan] * 10)  # full nan
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)

    pf = outlier_detector(pf, name)
    filtered_data = pf.data

    assert np.isnan(filtered_data[name]).all()

    # verify oddity filter still works as expected after receiving all nans
    values = np.array([1.0] * 10 + [5.0])  # 5 at the end is outlier
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)

    pf = outlier_detector(pf, name)
    filtered_data = pf.data

    assert not np.isnan(filtered_data[name].iloc[:10]).all()
    assert np.isnan(filtered_data[name].iloc[-1])  # filter outlier


def test_interspersed_nan_data(dummy_app_state: AppState):
    cfg = EMAFilterConfig(alpha=0.99, warmup=5)
    outlier_detector = EMAFilter(cfg, dummy_app_state)
    name = "sensor_x"

    values = np.array([np.nan, 1, 1, np.nan, 1, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1, 5, np.nan])
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)

    # feed oddity filter
    pf = outlier_detector(pf, name)

    expected = np.array([np.nan, 1, 1, np.nan, 1, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1, np.nan, np.nan])
    filtered_data = pf.data[name].to_numpy()
    assert np.allclose(filtered_data, expected, equal_nan=True)

    missing_info = pf.missing_info
    # should not mark existing nan as oddity
    assert missing_info["sensor_x"].iloc[0] == MissingType.NULL
    assert missing_info["sensor_x"].iloc[-2] == MissingType.OUTLIER


def test_streamed_nans(dummy_app_state: AppState):
    cfg = EMAFilterConfig(alpha=0.99, warmup=5)
    outlier_detector = EMAFilter(cfg, dummy_app_state)
    name = "sensor_x"

    values = np.array([np.nan, 1, 1, np.nan, 1, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1, 5, np.nan])
    expected = np.array([np.nan, 1, 1, np.nan, 1, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1, np.nan, np.nan])
    expected_missing_types = np.array([MissingType.NULL] * len(values))
    expected_missing_types[-2] = MissingType.OUTLIER

    ts = {}
    for in_val, expected_val, expected_missing_type in zip(values, expected, expected_missing_types, strict=True):
        data = pd.DataFrame({name: [in_val]})
        pf = PipelineFrame(data, DataMode.ONLINE, temporal_state=ts)

        # feed oddity filter
        pf = outlier_detector(pf, name)
        out_val = pf.data[name].to_numpy()[0]
        assert np.isclose(out_val, expected_val, equal_nan=True)

        missing_info = pf.missing_info
        assert missing_info[name].iloc[0] == expected_missing_type
        ts = pf.temporal_state


def test_obvious_outlier_in_first_batch(dummy_app_state: AppState):
    """
    The initial variance is calculated using the first batch,
    so we need a large batch in this test in order to ensure
    the outlier doesn't significantly poison the variance.

    If the online stats are initialized with outlier-free data,
    the outlier can be detected in a smaller subsequent batch
    """

    cfg = EMAFilterConfig(alpha=0.99)
    outlier_detector = EMAFilter(cfg, dummy_app_state)
    values = [1.0] * 1000
    values[-1] = 100.0  # <- this is the outlier
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)
    filtered_pf = outlier_detector(pf, name)
    filtered_data = filtered_pf.data

    assert not np.isnan(filtered_data[name].iloc[-2])
    assert np.isnan(filtered_data[name].iloc[-1])


def test_obvious_outlier_in_second_batch(dummy_app_state: AppState):
    """
    Note the batches here are smaller compared to the previous
    test (test_obvious_outlier_in_first_batch)
    """
    cfg = EMAFilterConfig(alpha=0.99)

    outlier_detector = EMAFilter(cfg, dummy_app_state)
    values = [1.0] * 10
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)
    pf = outlier_detector(pf, name)  # <- stats get initialized here

    values2 = [1.0] * 10
    values2[-1] = 100.0  # <- this is the outlier

    data2 = pd.DataFrame({name: values2})
    pf2 = PipelineFrame(data=data2, data_mode=DataMode.ONLINE, temporal_state=pf.temporal_state)
    filtered_pf2 = outlier_detector(pf2, name)
    filtered_data2 = filtered_pf2.data

    assert not np.isnan(filtered_data2[name].iloc[-2])
    assert np.isnan(filtered_data2[name].iloc[-1])


def test_obvious_outlier_in_stream(dummy_app_state: AppState):
    cfg = EMAFilterConfig(alpha=0.99)

    outlier_detector = EMAFilter(cfg, dummy_app_state)
    name = "sensor_x"

    values = [1.0]
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)

    for _ in range(10):
        pf = outlier_detector(pf, name)
        filtered_data = pf.data
        assert not np.isnan(filtered_data[name].iloc[0])

    # catch the outlier
    values = [10.0]  # <- this is the outlier
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data=data, data_mode=DataMode.ONLINE, temporal_state=pf.temporal_state)

    filtered_pf = outlier_detector(pf, name)
    filtered_data = filtered_pf.data

    assert np.isnan(filtered_data[name].iloc[0])


def test_detection_with_multiple_cols(dummy_app_state: AppState):
    cfg = EMAFilterConfig(alpha=0.99)

    outlier_detector_x = EMAFilter(cfg, dummy_app_state)
    outlier_detector_y = EMAFilter(cfg, dummy_app_state)
    name_x = "sensor_x"
    name_y = "sensor_y"

    values_x = [1.0]
    values_y = [2.0]
    data = pd.DataFrame({name_x: values_x, name_y: values_y})
    pf = PipelineFrame(data, DataMode.ONLINE)

    for _ in range(10):
        for tag, detector in zip([name_x, name_y], [outlier_detector_x, outlier_detector_y], strict=True):
            pf = detector(pf, tag)

        filtered_data = pf.data
        assert not filtered_data.isnull().values.any()

    # catch the outlier
    values_x = [10.0]  # <- this is the outlier
    values_y = [2.0]  # <- this is not an outlier

    data = pd.DataFrame({name_x: values_x, name_y: values_y})
    pf = PipelineFrame(data=data, data_mode=DataMode.ONLINE, temporal_state=pf.temporal_state)

    for tag, detector in zip([name_x, name_y], [outlier_detector_x, outlier_detector_y], strict=True):
        pf = detector(pf, tag)
    filtered_data = pf.data

    assert np.isnan(filtered_data[name_x].iloc[0])
    assert not np.isnan(filtered_data[name_y].iloc[0])


def test_detector_does_not_change_indices(dummy_app_state: AppState):
    cfg = EMAFilterConfig(alpha=0.99)

    outlier_detector = EMAFilter(cfg, dummy_app_state)
    n = 10
    values = [1.0] * n
    base_timestamp = datetime.strptime("31/01/24 23:59:59", "%d/%m/%y %H:%M:%S")
    timestamps = [base_timestamp + i * timedelta(minutes=5) for i in range(n)]
    name = "sensor_x"

    data = pd.DataFrame({name: values}, index=pd.DatetimeIndex(timestamps))
    pf = PipelineFrame(data, DataMode.ONLINE)

    pf = outlier_detector(pf, name)
    filtered_data = pf.data

    for i, dt_index in enumerate(filtered_data.index):
        ts = pd.to_datetime(dt_index)
        assert ts == timestamps[i]

    # catch the outlier
    values = [10.0]  # <- this is the outlier
    outlier_ts = timestamps[-1] + timedelta(minutes=5)
    data = pd.DataFrame({name: values}, index=pd.DatetimeIndex([outlier_ts]))
    pf = PipelineFrame(data=data, data_mode=DataMode.ONLINE, temporal_state=pf.temporal_state)

    filtered_pf = outlier_detector(pf, name)
    filtered_data = filtered_pf.data

    assert np.isnan(filtered_data[name].iloc[0])

    ts = pd.to_datetime(filtered_data.index[0])
    assert ts == outlier_ts


def test_outlier_gets_correct_missingtype(dummy_app_state: AppState):
    # initialize outlier detector
    cfg = EMAFilterConfig(alpha=0.99)

    # prepare some data to warm up the stats
    outlier_detector = EMAFilter(cfg, dummy_app_state)
    values = [1.0] * 10
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)
    outlier_detector(pf, name)  # <- stats get initialized here

    # create a batch with an outlier
    values2 = [1.0] * 10
    values2[-1] = 100.0  # <- this is the outlier

    data = pd.DataFrame({name: values2})
    pf = PipelineFrame(data=data, data_mode=DataMode.ONLINE, temporal_state=pf.temporal_state)

    # filter the outlier
    filtered_pf = outlier_detector(pf, name)

    # check that the outlier has the correct missing type
    missing_info = filtered_pf.missing_info
    assert missing_info["sensor_x"].iloc[-1] == MissingType.OUTLIER


def test_outlier_missing_type_is_added_to_existing_missing(dummy_app_state: AppState):
    # initialize outlier detector
    cfg = EMAFilterConfig(alpha=0.99)

    # prepare some data to warm up the stats
    outlier_detector = EMAFilter(cfg, dummy_app_state)
    values = [1.0] * 10
    name = "sensor_x"

    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)
    outlier_detector(pf, name)  # <- stats get initialized here

    # create a batch with an outlier
    values2 = [1.0] * 10
    values2[-1] = 100.0  # <- this is the outlier

    data = pd.DataFrame({name: values2})
    pf = PipelineFrame(data=data, data_mode=DataMode.ONLINE, temporal_state=pf.temporal_state)

    # add an initial missing type to the outlier
    pf.missing_info.loc[9, "sensor_x"] = MissingType.BOUNDS

    # filter the outlier
    filtered_pf = outlier_detector(pf, name)

    # check that the outlier has both missing types
    missing_info = filtered_pf.missing_info
    bitmap = MissingType(missing_info["sensor_x"].iloc[-1])

    assert bitmap == MissingType.BOUNDS | MissingType.OUTLIER
    assert MissingType.BOUNDS in bitmap
    assert MissingType.OUTLIER in bitmap

def test_filter_warmup_with_nans(dummy_app_state: AppState):
    name = "sensor_x"
    cfg = EMAFilterConfig(alpha=0.99, warmup=3)
    outlier_detector = EMAFilter(cfg, dummy_app_state)

    # first batch only has 2 non-NaN values, which shouldn't trigger outlier detection
    values = np.array([np.nan, 1, 1, np.nan, 5])
    data = pd.DataFrame({name: values})
    pf = PipelineFrame(data, DataMode.ONLINE)
    pf = outlier_detector(pf, name)
    filtered_data = pf.data[name].to_numpy()
    assert np.allclose(filtered_data, values, equal_nan=True)

    # completing warmup after the first value
    values2 = np.array([1, 5, np.nan, 5])  # the 5s should be detected as outliers
    data2 = pd.DataFrame({name: values2})
    pf2 = PipelineFrame(data2, DataMode.ONLINE, temporal_state=pf.temporal_state)
    pf2 = outlier_detector(pf2, name)

    filtered_data2 = pf2.data[name].to_numpy()
    expected2 = np.array([1, np.nan, np.nan, np.nan])  # Both 5s should be replaced with NaN
    assert np.allclose(filtered_data2, expected2, equal_nan=True)
