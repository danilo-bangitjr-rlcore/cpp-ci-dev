import datetime

import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.imputers.per_tag.copy import CopyImputer, CopyImputerConfig, CopyImputerTemporalState
from corerl.data_pipeline.missing_data_checker import missing_data_checker
from test.infrastructure.utils.pandas import dfs_close


def test_no_imputation():
    tag_1_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=1),
    )
    tag_2_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=2),
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


def test_all_nan_imputation():
    tag_1_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=1),
    )
    tag_2_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=2),
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

def test_all_nan_imputation_ts():
    tag_1_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=1),
    )
    tag_2_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=2),
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
            'tag_1': CopyImputerTemporalState(prev_horizon=4),
            'tag_2': CopyImputerTemporalState(prev_val=2.2),
        },
    }

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        'tag_1': [np.nan, np.nan, np.nan],
        'tag_2': [2.2, 2.2, np.nan],
    })

    assert dfs_close(pf.data, expected)
    ts = pf.temporal_state[StageCode.IMPUTER]
    assert ts is not None
    assert isinstance(ts, dict)

    tag_1_ts = ts['tag_1']
    assert tag_1_ts is not None
    assert isinstance(tag_1_ts, CopyImputerTemporalState)
    assert tag_1_ts.prev_val is not None
    assert np.isnan(tag_1_ts.prev_val)
    assert tag_1_ts.prev_horizon == 7

    tag_2_ts = ts['tag_2']
    assert tag_2_ts is not None
    assert isinstance(tag_2_ts, CopyImputerTemporalState)
    assert tag_2_ts.prev_val == 2.2
    assert tag_2_ts.prev_horizon == 3


def test_backtrack_imputation():
    tag_1_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=1),
    )
    tag_2_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=2),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, np.nan, np.nan, np.nan, 0.3, 0.1, np.nan],
        "tag_2": [-0.3, np.nan, 7.3, 1.2, np.nan, np.nan, np.nan, 2.2]
    })
    now = datetime.datetime.now(datetime.UTC)
    delta = datetime.timedelta(hours=1)
    indices = pd.Series([now, now+delta, now+2*delta, now+3*delta, now+4*delta, now+5*delta, now+6*delta, now+7*delta])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        'tag_1': [0.7, 0.7, 0.7, np.nan, 0.3, 0.3, 0.1, 0.1],
        'tag_2': [-0.3, -0.3, 7.3, 1.2, 1.2, 1.2, 2.2, 2.2],
    })
    assert dfs_close(pf.data, expected)


def test_backtrack_imputation_ts():
    tag_1_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=1),
    )
    tag_2_imputer = CopyImputer(
        CopyImputerConfig(imputation_horizon=2),
    )

    data = pd.DataFrame({
        "tag_1": [np.nan, 0.7, np.nan],
        "tag_2": [np.nan, 6.6, np.nan]
    })
    indices = pd.Series(["7/13/2023 10:00", "7/13/2023 11:00", "7/13/2023 12:00"])
    indices = pd.to_datetime(indices)
    data = data.set_index(indices)

    pf = PipelineFrame(data, CallerCode.ONLINE)

    # Create ImputerTemporalState
    pf.temporal_state = {
        StageCode.IMPUTER: {
            'tag_1': CopyImputerTemporalState(prev_val=1.0),
            'tag_2': CopyImputerTemporalState(prev_val=2.0, prev_horizon=2),
        },
    }

    # Start data pipeline
    pf = missing_data_checker(pf, 'tag_1')
    pf = missing_data_checker(pf, 'tag_2')

    pf = tag_1_imputer(pf, 'tag_1')
    pf = tag_2_imputer(pf, 'tag_2')

    expected = pd.DataFrame({
        'tag_1': [1.0, 0.7, 0.7],
        'tag_2': [6.6, 6.6, 6.6],
    })
    assert dfs_close(pf.data, expected)
