import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.imputers.per_tag.linear import LinearImputer, LinearImputerConfig, LinearImputerTemporalState
from test.infrastructure.utils.pandas import dfs_close


def test_no_imputation():
    """
    1. Test that pf.data remains unchanged when there are no NaNs
    2. Ensure the temporal states produced by the imputer set 'prev_val' to the last non-NaN value
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [0.2, 0.3, 0.7],
        "tag_2": [1.9, -0.5, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    # with no missing data, expect imputation to
    # be a no-op
    assert dfs_close(pf.data, data)
    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val == 0.7
    assert tag_1_ts.num_nans == 0

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val == 7.3
    assert tag_2_ts.num_nans == 0

def test_all_nan_imputation():
    """
    1. Test that pf.data remains unchanged when it consists of all NaNs
    2. Ensure the temporal states produced by the imputer have NaNs as their 'prev_val's
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, np.nan],
        "tag_2": [np.nan, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isnan(pf.data["tag_1"]).all()
    assert np.isnan(pf.data["tag_2"]).all()

    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val is not None
    assert np.isnan(tag_1_ts.prev_val)
    assert tag_1_ts.num_nans == 0

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val is not None
    assert np.isnan(tag_2_ts.prev_val)
    assert tag_2_ts.num_nans == 0

def test_all_nan_imputation_ts():
    """
    1. Test that pf.data remains all NaNs - even if there are non-NaN values in the temporal state
    2. Ensure the temporal states produced by the imputer have NaNs as their 'prev_val's
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, np.nan],
        "tag_2": [np.nan, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create temporal state
    pf.temporal_state = {
        StageCode.IMPUTER: {
            'tag_1': LinearImputerTemporalState(prev_val=np.nan, num_nans=0),
            'tag_2': LinearImputerTemporalState(prev_val=8.0, num_nans=1),
        },
    }

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    assert np.isnan(pf.data["tag_1"]).all()
    assert np.isnan(pf.data["tag_2"]).all()

    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val is not None
    assert np.isnan(tag_1_ts.prev_val)
    assert tag_1_ts.num_nans == 0

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val is not None
    assert np.isnan(tag_2_ts.prev_val)
    assert tag_2_ts.num_nans == 0

def test_leading_nan_imputation():
    """
    1. Ensure the leading NaNs in pf.data remain NaNs
    2. Ensure the temporal states produced by the imputer set 'prev_val' to the last non-NaN value
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, 0.8],
        "tag_2": [np.nan, np.nan, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        'tag_1': [np.nan, 0.7, 0.8],
        'tag_2': [np.nan, np.nan, 7.3],
    })

    assert dfs_close(pf.data, expected)
    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val == 0.8
    assert tag_1_ts.num_nans == 0

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val == 7.3
    assert tag_2_ts.num_nans == 0

def test_leading_nan_imputation_ts():
    """
    1. Ensure the leading NaNs in pf.data remain NaNs when the temporal state's 'prev_val' is a NaN
    2. Ensure the leading NaNs are imputed when the first non-NaN value in the series
    is within 'max_gap' of the non-NaN value in the temporal state
    3. Ensure the temporal states produced by the imputer set 'prev_val' to the last non-NaN value
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, 0.8],
        "tag_2": [np.nan, np.nan, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create temporal state
    pf.temporal_state = {
        StageCode.IMPUTER: {
            'tag_1': LinearImputerTemporalState(prev_val=np.nan, num_nans=0),
            'tag_2': LinearImputerTemporalState(prev_val=1.3, num_nans=0),
        },
    }

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        'tag_1': [np.nan, 0.7, 0.8],
        'tag_2': [3.3, 5.3, 7.3],
    })

    assert dfs_close(pf.data, expected)
    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val == 0.8
    assert tag_1_ts.num_nans == 0

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val == 7.3
    assert tag_2_ts.num_nans == 0

def test_trailing_nan_imputation():
    """
    1. Ensure the trailing NaNs in pf.data remain NaNs
    2. Ensure the temporal states produced by the imputer set 'prev_val' to the last non-NaN value
    within 'max_gap' of the end of the series
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [0.2, 0.7, np.nan],
        "tag_2": [0.8, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        'tag_1': [0.2, 0.7, np.nan],
        'tag_2': [0.8, np.nan, np.nan],
    })

    assert dfs_close(pf.data, expected)
    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val == 0.7
    assert tag_1_ts.num_nans == 1

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val == 0.8
    assert tag_2_ts.num_nans == 2

def test_trailing_nan_imputation_ts():
    """
    1. Ensure the trailing NaNs in pf.data remain NaNs despite there being data in the temporal state
    2. Ensure the temporal states produced by the imputer set 'prev_val' to the last non-NaN value
    within 'max_gap' of the end of the series
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [0.2, 0.7, np.nan],
        "tag_2": [0.8, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create temporal state
    pf.temporal_state = {
        StageCode.IMPUTER: {
            'tag_1': LinearImputerTemporalState(prev_val=np.nan, num_nans=0),
            'tag_2': LinearImputerTemporalState(prev_val=0.9, num_nans=1),
        },
    }

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        'tag_1': [0.2, 0.7, np.nan],
        'tag_2': [0.8, np.nan, np.nan],
    })

    assert dfs_close(pf.data, expected)
    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val == 0.7
    assert tag_1_ts.num_nans == 1

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val == 0.8
    assert tag_2_ts.num_nans == 2

def test_linear_interpolation():
    """
    1. Ensure NaNs are imputed when they are between two non-NaN values that are separated by at most 'max_gap'
    2. Ensure the temporal states produced by the imputer set 'prev_val' to the last non-NaN value
    within 'max_gap' of the end of the series
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, np.nan, 0.9, np.nan],
        "tag_2": [np.nan, 0.2, np.nan, np.nan, 0.8]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00", "7/13/2023 13:00", "7/13/2023 14:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        "tag_1": [np.nan, 0.7, 0.8, 0.9, np.nan],
        "tag_2": [np.nan, 0.2, 0.4, 0.6, 0.8]
    })

    assert dfs_close(pf.data, expected)
    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val == 0.9
    assert tag_1_ts.num_nans == 1

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val == 0.8
    assert tag_2_ts.num_nans == 0

def test_linear_interpolation_ts():
    """
    1. Ensure NaNs are imputed when they are between two non-NaN values that are separated by at most 'max_gap'.
    This includes leading NaNs that may now be imputed due to there being a non-NaN value in the temporal state
    2. Ensure the temporal states produced by the imputer set 'prev_val' to the last non-NaN value
    within 'max_gap' of the end of the series
    """
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, np.nan, 0.9, np.nan],
        "tag_2": [np.nan, 0.2, np.nan, np.nan, 0.8]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00", "7/13/2023 13:00", "7/13/2023 14:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create temporal state
    pf.temporal_state = {
        StageCode.IMPUTER: {
            'tag_1': LinearImputerTemporalState(prev_val=0.5, num_nans=0),
            'tag_2': LinearImputerTemporalState(prev_val=-0.4, num_nans=1),
        },
    }

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        "tag_1": [0.6, 0.7, 0.8, 0.9, np.nan],
        "tag_2": [0.0, 0.2, 0.4, 0.6, 0.8]
    })

    assert dfs_close(pf.data, expected)
    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, LinearImputerTemporalState)
    assert tag_1_ts.prev_val == 0.9
    assert tag_1_ts.num_nans == 1

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, LinearImputerTemporalState)
    assert tag_2_ts.prev_val == 0.8
    assert tag_2_ts.num_nans == 0
