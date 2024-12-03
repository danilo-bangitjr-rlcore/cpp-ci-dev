import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import PipelineFrame, StageCode, CallerCode
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from corerl.data_pipeline.imputers.linear import LinearImputer, LinearImputerConfig, LinearImputerTemporalState
from corerl.data_pipeline.tag_config import TagConfig
from test.infrastructure.utils.pandas import dfs_close

def test_no_imputation():
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
    )

    data = pd.DataFrame({
        "tag_1": [0.2, 0.3, 0.7],
        "tag_2": [1.9, -0.5, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, np.nan, np.nan],
        "tag_2": [np.nan, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
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
            'tag_2': LinearImputerTemporalState(prev_val=np.nan, num_nans=0),
        },
    }

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, 0.8],
        "tag_2": [np.nan, np.nan, 7.3]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
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
            'tag_2': LinearImputerTemporalState(prev_val=np.nan, num_nans=0),
        },
    }

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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

def test_trailing_nan_imputation():
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
    )

    data = pd.DataFrame({
        "tag_1": [0.2, 0.7, np.nan],
        "tag_2": [0.8, np.nan, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
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

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, np.nan, 0.9, np.nan],
        "tag_2": [np.nan, 0.2, np.nan, np.nan, 0.8]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00", "7/13/2023 13:00", "7/13/2023 14:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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
    tag_1_imputer = LinearImputer(
        LinearImputerConfig(max_gap=1),
        TagConfig(name="tag_1", bounds=(0.0, 1.0)),
    )
    tag_2_imputer = LinearImputer(
        LinearImputerConfig(max_gap=2),
        TagConfig(name="tag_2", bounds=(-1.0, 10.0)),
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

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

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
