import numpy as np
import pandas as pd
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.state_constructors.components.norm import NormalizerConfig
from corerl.data_pipeline.state_constructors.components.add_raw import AddRawConfig
from corerl.data_pipeline.state_constructors.components.split import SplitConfig
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


def test_sc_add_raw():
    raw_obs = pd.DataFrame({
        'tag_1': [np.nan, 1, 2, 3, np.nan, 1, 2],
        'tag_2': [1, 2, 3, np.nan, 1, 2, np.nan],
    })

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    sc = StateConstructor(
        cfgs=[
            TraceConfig(trace_values=[0.1, 0.01]),
            AddRawConfig(),
        ]
    )

    for tag_name in ('tag_1', 'tag_2'):
        pf = sc(pf, tag_name)

    expected = pd.DataFrame({
        'tag_1':            [np.nan, 1, 2, 3, np.nan, 1, 2],
        'tag_2':            [1, 2, 3, np.nan, 1, 2, np.nan],
        'tag_1_trace-0.1':  [np.nan, 1., 1.9, 2.89, np.nan, 1., 1.9],
        'tag_1_trace-0.01': [np.nan, 1., 1.99, 2.9899, np.nan, 1., 1.99],
        'tag_2_trace-0.1':  [1., 1.9, 2.89, np.nan, 1., 1.9, np.nan],
        'tag_2_trace-0.01': [1., 1.99, 2.9899, np.nan, 1., 1.99, np.nan],
    })
    assert dfs_close(pf.data, expected)


def test_sc_integration1():
    """
    Tests the integration of several state-constructor parts into a standard
    expected state constructor pattern. Expected outcome is that we obtain
    normalized traces and the raw value of the tag *and* the normalized value
    of the tag (because add_raw does not consume any prior values).
    """
    raw_obs = pd.DataFrame({
        'tag-1': [np.nan, 0, 1, 2, 3, 4, 5, np.nan, 1, 2],
    })

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.OFFLINE,
    )

    sc = StateConstructor(
        cfgs=[
            NormalizerConfig(),
            SplitConfig(
                left=TraceConfig(trace_values=[0.1]),
                right=AddRawConfig(),
            ),
        ],
    )

    pf = sc(pf, 'tag-1')

    expected = pd.DataFrame({
        'tag-1':                [np.nan, 0, 1, 2, 3, 4, 5, np.nan, 1, 2],
        'tag-1_norm':           [np.nan, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan, 0.2, 0.4],
        'tag-1_norm_trace-0.1': [np.nan, 0, 0.18, 0.378, 0.5778, 0.77778, 0.977778, np.nan, 0.2, 0.38],
    })
    assert dfs_close(pf.data, expected)


def test_sc_integration2():
    """
    Tests the integration of several state-constructor parts into a standard
    expected state constructor pattern. Expected outcome is that we obtain
    normalized traces and the raw value of the tag.
    """
    raw_obs = pd.DataFrame({
        'tag-1': [np.nan, 0, 1, 2, 3, 4, 5, np.nan, 1, 2],
    })

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.OFFLINE,
    )

    sc = StateConstructor(
        cfgs=[
            NormalizerConfig(),
            SplitConfig(
                left=TraceConfig(trace_values=[0.1]),
                right=AddRawConfig(),
                passthrough=False
            ),
        ],
    )

    pf = sc(pf, 'tag-1')

    expected = pd.DataFrame({
        'tag-1':                [np.nan, 0, 1, 2, 3, 4, 5, np.nan, 1, 2],
        'tag-1_norm_trace-0.1': [np.nan, 0, 0.18, 0.378, 0.5778, 0.77778, 0.977778, np.nan, 0.2, 0.38],
    })
    assert dfs_close(pf.data, expected)


def test_sc_integration3():
    """
    Tests the integration of several state-constructor parts into a standard
    expected state constructor pattern. Expected outcome is that we obtain
    normalized traces *and* the normalized value.
    """
    raw_obs = pd.DataFrame({
        'tag-1': [np.nan, 0, 1, 2, 3, 4, 5, np.nan, 1, 2],
    })

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.OFFLINE,
    )

    sc = StateConstructor(
        cfgs=[
            NormalizerConfig(),
            SplitConfig(
                left=TraceConfig(trace_values=[0.1]),
                right=AddRawConfig(),
                passthrough=True,
            ),
        ],
    )

    pf = sc(pf, 'tag-1')

    expected = pd.DataFrame({
        'tag-1':                 [np.nan, 0, 1, 2, 3, 4, 5, np.nan, 1, 2],
        'tag-1_norm':            [np.nan, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan, 0.2, 0.4],
        'tag-1_norm_trace-0.1':  [np.nan, 0, 0.18, 0.378, 0.5778, 0.77778, 0.977778, np.nan, 0.2, 0.38],
    })
    assert dfs_close(pf.data, expected)

def test_sc_integration4():
    """
    Tests the integration of several state-constructor parts into a standard
    expected state constructor pattern. Expected outcome is that we obtain
    normalized traces *and* the normalized value.
    """
    raw_obs = pd.DataFrame({
        'tag-1': [np.nan, 0, 1, 2, 3, 4, 5, np.nan, 1, 2],
    })

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.OFFLINE,
    )

    sc = StateConstructor(
        cfgs=[
            NormalizerConfig(),
            SplitConfig(
                left=TraceConfig(trace_values=[0.1]),
                right=TraceConfig(trace_values=[0.01]),
                passthrough=True,
            ),
        ],
    )

    pf = sc(pf, 'tag-1')

    expected = pd.DataFrame({
        'tag-1_norm':             [np.nan, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan, 0.2, 0.4],
        'tag-1_norm_trace-0.1':   [np.nan, 0, 0.18, 0.378, 0.5778, 0.77778, 0.977778, np.nan, 0.2, 0.38],
        'tag-1_norm_trace-0.01':  [np.nan, 0, 0.198, 0.39798, 0.59798, 0.79798, 0.99798, np.nan, 0.2, 0.398],
    })
    assert dfs_close(pf.data, expected)
