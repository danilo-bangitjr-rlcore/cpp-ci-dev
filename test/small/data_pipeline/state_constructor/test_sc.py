import numpy as np
import pandas as pd
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.state_constructors.sc import StateConstructor
from corerl.data_pipeline.state_constructors.components.trace import TraceConfig

def test_sc1():
    raw_obs = pd.DataFrame({
        'obs_1': [np.nan, 1, 2, 3, np.nan, 1, 2],
        'obs_2': [1, 2, 3, np.nan, 1, 2, np.nan],
    })

    pf = PipelineFrame(
        data=raw_obs,
    )

    sc = StateConstructor(
        cfgs=[
            TraceConfig(trace_values=[0.1, 0.01]),
        ]
    )

    for tag_name in ('obs_1', 'obs_2'):
        pf = sc(pf, tag_name)

    expected_data = pd.DataFrame({
        'obs_1_trace-0.1':  [np.nan, 1., 1.9, 2.89, np.nan, 1., 1.9],
        'obs_1_trace-0.01': [np.nan, 1., 1.99, 2.9899, np.nan, 1., 1.99],
        'obs_2_trace-0.1':  [1., 1.9, 2.89, np.nan, 1., 1.9, np.nan],
        'obs_2_trace-0.01': [1., 1.99, 2.9899, np.nan, 1., 1.99, np.nan],
    })

    assert _dfs_close(pf.data, expected_data)



def _dfs_close(df1: pd.DataFrame, df2: pd.DataFrame):
    if set(df1.columns) != set(df2.columns):
        return False

    for col in df1.columns:
        if not np.allclose(df1[col], df2[col], equal_nan=True):
            return False

    return True
