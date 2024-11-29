import numpy as np
import pandas as pd
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.state_constructors.components.norm import NormalizerConfig
from corerl.data_pipeline.state_constructors.sc import StateConstructor
from corerl.data_pipeline.state_constructors.components.trace import TraceConfig

from test.infrastructure.utils.pandas import dfs_close

def test_sc1():
    raw_obs = pd.DataFrame({
        'obs_1': [np.nan, 1, 2, 3, np.nan, 1, 2],
        'obs_2': [1, 2, 3, np.nan, 1, 2, np.nan],
    })

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
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

    assert dfs_close(pf.data, expected_data)

def test_norm_sc():
    obs = pd.DataFrame({
        'tag-1': [np.nan, 0, 1, 2, 3, 4, np.nan, 1, 2],
    })
    pf = PipelineFrame(
        data=obs,
        caller_code=CallerCode.OFFLINE,
    )

    sc = StateConstructor(
        cfgs=[
            NormalizerConfig(),
            TraceConfig(trace_values=[0.1, 0.01]),
        ]
    )
    pf = sc(pf, 'tag-1')

    expected = pd.DataFrame({
        'tag-1_norm_trace-0.1':  [np.nan, 0, .225, .4725, .72225, .972225, np.nan, .25, .475],
        'tag-1_norm_trace-0.01': [np.nan, 0, .2475, .497475, .747475, .997475, np.nan, .25, .4975],
    })
    assert dfs_close(pf.data, expected)
