import numpy as np
import pandas as pd
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.add_raw import AddRawConfig
from corerl.data_pipeline.transforms.split import SplitConfig
from corerl.data_pipeline.state_constructors.sc import SCConfig, StateConstructor
from corerl.data_pipeline.transforms.trace import TraceConfig

from corerl.data_pipeline.tag_config import TagConfig
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
        tag_cfgs=[
            TagConfig(name='obs_1'),
            TagConfig(name='obs_2'),
        ],
        cfg=SCConfig(
            defaults=[
                TraceConfig(trace_values=[0.1, 0.01]),
            ],
            countdown=CountdownConfig(action_period=1),
        ),
    )

    pf = sc(pf)

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
        tag_cfgs=[
            TagConfig(
                name='tag-1',
                state_constructor=[
                    NormalizerConfig(),
                    TraceConfig(trace_values=[0.1, 0.01]),
                ],
            ),
        ],
        cfg=SCConfig(
            defaults=[],
            countdown=CountdownConfig(action_period=1),
        ),
    )
    pf = sc(pf)

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
        tag_cfgs=[
            TagConfig(name='tag_1'),
            TagConfig(name='tag_2'),
        ],
        cfg=SCConfig(
            defaults=[
                TraceConfig(trace_values=[0.1, 0.01]),
                AddRawConfig(),
            ],
            countdown=CountdownConfig(action_period=1),
        ),
    )

    pf = sc(pf)
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
        tag_cfgs=[
            TagConfig(name='tag-1'),
        ],
        cfg=SCConfig(
            defaults=[
                NormalizerConfig(),
                SplitConfig(
                    left=[TraceConfig(trace_values=[0.1])],
                    right=[AddRawConfig()],
                ),
            ],
            countdown=CountdownConfig(action_period=1),
        ),
    )

    pf = sc(pf)

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
        tag_cfgs=[
            TagConfig(
                name='tag-1',
                state_constructor=[
                    NormalizerConfig(),
                    SplitConfig(
                        left=[TraceConfig(trace_values=[0.1])],
                        right=[AddRawConfig()],
                        passthrough=False
                    ),
                ],
            ),
        ],
        cfg=SCConfig(
            defaults=[],
            countdown=CountdownConfig(action_period=1),
        ),
    )

    pf = sc(pf)

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
        tag_cfgs=[
            TagConfig(name='tag-1'),
        ],
        cfg=SCConfig(
            defaults=[
                NormalizerConfig(),
                SplitConfig(
                    left=[TraceConfig(trace_values=[0.1])],
                    right=[AddRawConfig()],
                    passthrough=True,
                ),
            ],
            countdown=CountdownConfig(action_period=1),
        ),
    )

    pf = sc(pf)

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
        tag_cfgs=[
            TagConfig(name='tag-1'),
        ],
        cfg=SCConfig(
            defaults=[
                NormalizerConfig(),
                SplitConfig(
                    left=[TraceConfig(trace_values=[0.1])],
                    right=[TraceConfig(trace_values=[0.01])],
                    passthrough=True,
                ),
            ],
            countdown=CountdownConfig(action_period=1),
        ),
    )

    pf = sc(pf)

    expected = pd.DataFrame({
        'tag-1_norm':             [np.nan, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan, 0.2, 0.4],
        'tag-1_norm_trace-0.1':   [np.nan, 0, 0.18, 0.378, 0.5778, 0.77778, 0.977778, np.nan, 0.2, 0.38],
        'tag-1_norm_trace-0.01':  [np.nan, 0, 0.198, 0.39798, 0.59798, 0.79798, 0.99798, np.nan, 0.2, 0.398],
    })
    assert dfs_close(pf.data, expected)


def test_sc_decision_point_detection():
    """
    Tests the integration of the decision point detector
    and the broader state constructor. Expect action interrupts
    and the action period to both produce valid decision points
    and the sc transforms non-action tags.
    """
    raw_obs = pd.DataFrame({
        'tag-1': [np.nan, 0, 1, 2, 3, 4, 5, np.nan, 1, 2],
        'tag-action': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.OFFLINE,
    )

    sc = StateConstructor(
        tag_cfgs=[
            TagConfig(name='tag-1'),
            TagConfig(name='tag-action', tag_type="action"),
        ],
        cfg=SCConfig(
            defaults=[
                NormalizerConfig(),
            ],
            countdown=CountdownConfig(
                kind='int',
                action_period=4,
            ),
        ),
    )

    pf = sc(pf)

    expected = pd.DataFrame({
        'tag-1_norm':    [np.nan, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan, 0.2, 0.4],
        'tag-action':    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        'countdown.[0]': [3, 2, 1, 4, 3, 2, 1, 4, 3, 2],
    })
    assert dfs_close(pf.data, expected)
    assert np.all(
        pf.decision_points == np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    )


def test_per_tag_overrides():
    raw_obs = pd.DataFrame({
        'tag_1': [np.nan, 1, 2, 3, np.nan, 1, 2],
        'tag_2': [1, 2, 3, np.nan, 1, 2, np.nan],
    })

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    sc = StateConstructor(
        tag_cfgs=[
            TagConfig(name='tag_1'),
            TagConfig(
                name='tag_2',
                state_constructor=[
                    TraceConfig(trace_values=[0.1])
                ]
            ),
        ],
        cfg=SCConfig(
            defaults=[
                TraceConfig(trace_values=[0.1, 0.01]),
                AddRawConfig(),
            ],
            countdown=CountdownConfig(action_period=4),
        ),
    )

    pf = sc(pf)
    expected = pd.DataFrame({
        'tag_1':            [np.nan, 1, 2, 3, np.nan, 1, 2],
        'tag_1_trace-0.1':  [np.nan, 1., 1.9, 2.89, np.nan, 1., 1.9],
        'tag_1_trace-0.01': [np.nan, 1., 1.99, 2.9899, np.nan, 1., 1.99],
        'tag_2_trace-0.1':  [1., 1.9, 2.89, np.nan, 1., 1.9, np.nan],
    })
    assert dfs_close(pf.data, expected)
