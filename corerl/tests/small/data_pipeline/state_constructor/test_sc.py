from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from lib_defs.config_defs.tag_config import TagType
from test.infrastructure.utils.pandas import dfs_close

from corerl.data_pipeline.constructors.sc import SCConfig, StateConstructor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.transforms.add_raw import AddRawConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.split import SplitConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.state import AppState
from corerl.tags.tag_config import BasicTagConfig


@pytest.fixture
def test_pf():
    raw_obs = pd.DataFrame({
        'tag_1': [1, 2, 3],
        'tag_2': [1, 2, 3],
        'action': [0, 0, 0],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1],
    })

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.REFRESH,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi
    return pf

def test_sc1(test_pf: PipelineFrame, dummy_app_state: AppState):
    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(name='tag_1'),
            BasicTagConfig(name='tag_2'),
            BasicTagConfig(name='action'),
        ],
        cfg=SCConfig(
            defaults=[
                TraceConfig(trace_values=[0.1, 0.01], missing_tol=1.0),
            ],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )

    pf = sc(test_pf)

    expected_data = pd.DataFrame({
        'tag_1_trace-0.1':   [1., 1.9, 2.89],
        'tag_1_trace-0.01':  [1., 1.99, 2.9899],
        'tag_2_trace-0.1':   [1., 1.9, 2.89],
        'tag_2_trace-0.01':  [1., 1.99, 2.9899],
        'action_trace-0.1': [0., 0., 0.],
        'action_trace-0.01': [0., 0., 0.],
        'action-lo': [0, 0, 0],
        'action-hi': [1, 1, 1],
    })

    assert dfs_close(pf.data, expected_data)

def test_norm_sc(dummy_app_state: AppState):
    obs = pd.DataFrame({
        'tag-1': [0, 1, 2, 3, 4],
        'action': [0, 0, 0, 0, 0],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=obs,
        data_mode=DataMode.OFFLINE,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(
                name='tag-1',
                state_constructor=[
                    NormalizerConfig(from_data=True),
                    TraceConfig(trace_values=[0.1, 0.01], missing_tol=1.0),
                ],
            ),
            BasicTagConfig(
                name='action',
                state_constructor=[
                    NormalizerConfig(
                        from_data=False,
                        min=0.,
                        max=1.,
                    ),
                ],
            ),
        ],
        cfg=SCConfig(
            defaults=[],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )
    pf = sc(pf)
    print("Output:")
    print(pf.data.to_string())


    expected = pd.DataFrame({
        'tag-1_norm_trace-0.1':  [0, .225, .4725, .72225, .972225],
        'tag-1_norm_trace-0.01': [0, .2475, .497475, .747475, .997475],
        'action_norm':           [0, 0, 0, 0, 0],
        'action-lo':             [0, 0, 0, 0, 0],
        'action-hi':             [1, 1, 1, 1, 1],
    })
    assert dfs_close(pf.data, expected)


def test_sc_add_raw(test_pf: PipelineFrame, dummy_app_state: AppState):
    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(name='tag_1'),
            BasicTagConfig(name='tag_2'),
            BasicTagConfig(name='action'),
        ],
        cfg=SCConfig(
            defaults=[
                TraceConfig(trace_values=[0.1, 0.01]),
                AddRawConfig(),
            ],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )

    pf = sc(test_pf)
    expected = pd.DataFrame({
        'tag_1':             [ 1, 2, 3],
        'tag_2':             [1, 2, 3],
        'action':            [0, 0, 0],
        'tag_1_trace-0.1':   [1., 1.9, 2.89],
        'tag_1_trace-0.01':  [1., 1.99, 2.9899],
        'tag_2_trace-0.1':   [1., 1.9, 2.89],
        'tag_2_trace-0.01':  [1., 1.99, 2.9899],
        'action_trace-0.1': [0., 0., 0.],
        'action_trace-0.01': [0., 0., 0.],
        'action-lo': [0, 0, 0],
        'action-hi': [1, 1, 1],
    })
    assert dfs_close(pf.data, expected)


def test_sc_integration1(dummy_app_state: AppState):
    """
    Tests the integration of several state-constructor parts into a standard
    expected state constructor pattern. Expected outcome is that we obtain
    normalized traces and the raw value of the tag *and* the normalized value
    of the tag (because add_raw does not consume any prior values).
    """
    raw_obs = pd.DataFrame({
        'tag-1': [0, 1, 2, 3, 4, 5],
        'action': [0, 1, 0, 1, 0, 1],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0, 0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.OFFLINE,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(name='tag-1'),
            BasicTagConfig(name='action'),
        ],
        cfg=SCConfig(
            defaults=[
                NormalizerConfig(from_data=True),
                SplitConfig(
                    left=[TraceConfig(trace_values=[0.1], missing_tol=1.0)],
                    right=[AddRawConfig()],
                ),
            ],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )

    pf = sc(pf)

    expected = pd.DataFrame({
        'tag-1':                 [0, 1, 2, 3, 4, 5],
        'tag-1_norm':            [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'tag-1_norm_trace-0.1':  [0, 0.18, 0.378, 0.5778, 0.77778, 0.977778],
        'action':                [0, 1, 0, 1, 0, 1],
        'action_norm':           [0, 1, 0, 1, 0, 1],
        'action_norm_trace-0.1': [0, 0.9, 0.09, 0.909, 0.0909, 0.90909],
        'action-lo':             [0, 0, 0, 0, 0, 0],
        'action-hi':             [1, 1, 1, 1, 1, 1],
    })
    assert dfs_close(pf.data, expected)


def test_sc_integration2(dummy_app_state: AppState):
    """
    Tests the integration of several state-constructor parts into a standard
    expected state constructor pattern. Expected outcome is that we obtain
    normalized traces and the raw value of the tag.
    """
    raw_obs = pd.DataFrame({
        'tag-1': [0, 1, 2, 3, 4, 5],
        'action': [0, 0, 0, 0, 0, 0],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0, 0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.OFFLINE,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(
                name='tag-1',
                state_constructor=[
                    NormalizerConfig(from_data=True),
                    SplitConfig(
                        left=[TraceConfig(trace_values=[0.1], missing_tol=1.0)],
                        right=[AddRawConfig()],
                        passthrough=False,
                    ),
                ],
            ),
            BasicTagConfig(
                name='action',
                state_constructor=[
                    NormalizerConfig(
                        from_data=False,
                        min=0.,
                        max=1.,
                    ),
                    SplitConfig(
                        left=[TraceConfig(trace_values=[0.1], missing_tol=1.0)],
                        right=[AddRawConfig()],
                        passthrough=False,
                    ),
                ],
            ),
        ],
        cfg=SCConfig(
            defaults=[],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )

    pf = sc(pf)
    print("Output:")
    print(pf.data.to_string())

    expected = pd.DataFrame({
        'tag-1':                 [0, 1, 2, 3, 4, 5],
        'tag-1_norm_trace-0.1':  [0, 0.18, 0.378, 0.5778, 0.77778, 0.977778],
        'action':                [0, 0, 0, 0, 0, 0],
        'action_norm_trace-0.1': [0, 0, 0, 0, 0, 0],
        'action-lo':             [0, 0, 0, 0, 0, 0],
        'action-hi':             [1, 1, 1, 1, 1, 1],
    })
    assert dfs_close(pf.data, expected)


def test_sc_integration3(dummy_app_state: AppState):
    """
    Tests the integration of several state-constructor parts into a standard
    expected state constructor pattern. Expected outcome is that we obtain
    normalized traces *and* the normalized value.
    """
    raw_obs = pd.DataFrame({
        'tag-1': [0, 1, 2, 3, 4, 5],
        'action': [0, 0, 0, 0, 0, 1],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0, 0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.OFFLINE,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(name='tag-1'),
            BasicTagConfig(name='action'),
        ],
        cfg=SCConfig(
            defaults=[
                NormalizerConfig(from_data=True),
                SplitConfig(
                    left=[TraceConfig(trace_values=[0.1], missing_tol=1.0)],
                    right=[AddRawConfig()],
                    passthrough=True,
                ),
            ],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )

    pf = sc(pf)

    expected = pd.DataFrame({
        'tag-1':                 [0, 1, 2, 3, 4, 5],
        'tag-1_norm':            [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'tag-1_norm_trace-0.1':  [0, 0.18, 0.378, 0.5778, 0.77778, 0.977778],
        'action':                [0, 0, 0, 0, 0, 1],
        'action_norm':           [0, 0, 0, 0, 0, 1],
        'action_norm_trace-0.1': [0, 0, 0, 0, 0, 0.9],
        'action-lo':             [0, 0, 0, 0, 0, 0],
        'action-hi':             [1, 1, 1, 1, 1, 1],
    })
    assert dfs_close(pf.data, expected)

def test_sc_integration4(dummy_app_state: AppState):
    """
    Tests the integration of several state-constructor parts into a standard
    expected state constructor pattern. Expected outcome is that we obtain
    normalized traces *and* the normalized value.
    """
    raw_obs = pd.DataFrame({
        'tag-1': [0, 1, 2, 3, 4, 5],
        'action': [0, 0, 0, 0, 0, 1],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0, 0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.OFFLINE,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(name='tag-1'),
            BasicTagConfig(name='action'),
        ],
        cfg=SCConfig(
            defaults=[
                NormalizerConfig(from_data=True),
                SplitConfig(
                    left=[TraceConfig(trace_values=[0.1], missing_tol=1.0)],
                    right=[TraceConfig(trace_values=[0.01], missing_tol=1.0)],
                    passthrough=True,
                ),
            ],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )

    pf = sc(pf)
    print("Output:")
    print(pf.data.to_string())

    expected = pd.DataFrame({
        'tag-1_norm':             [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'tag-1_norm_trace-0.1':   [0, 0.18, 0.378, 0.5778, 0.77778, 0.977778],
        'tag-1_norm_trace-0.01':  [0, 0.198, 0.39798, 0.59798, 0.79798, 0.99798],
        'action_norm':            [0, 0, 0, 0, 0, 1],
        'action_norm_trace-0.1':  [0, 0, 0, 0, 0, 0.9],
        'action_norm_trace-0.01': [0, 0, 0, 0, 0, 0.99],
        'action-lo':              [0, 0, 0, 0, 0, 0],
        'action-hi':              [1, 1, 1, 1, 1, 1],
    })
    assert dfs_close(pf.data, expected)


def test_sc_decision_point_detection(dummy_app_state: AppState):
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

    raw_action = pd.DataFrame(raw_obs['tag-action'])
    action_lo = pd.DataFrame({
        'tag-action-lo': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'tag-action-hi': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.OFFLINE,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    # stub out action construction
    pf.actions = raw_action

    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(name='tag-1'),
            BasicTagConfig(name='tag-action', state_constructor=[], type=TagType.ai_setpoint),
        ],
        cfg=SCConfig(
            defaults=[
                NormalizerConfig(from_data=True),
            ],
            countdown=CountdownConfig(
                kind='int',
                action_period=timedelta(minutes=4),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )

    pf = sc(pf)

    expected = pd.DataFrame({
        'countdown.[0]': [2,      1,   0,   3,   2,   1,   0,      3,   2,   1],
        'tag-1_norm':    [np.nan, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan, 0.2, 0.4],
        'tag-action':    [0,      0,   0,   1,   1,   1,   1,      1,   1,   1],
        'tag-action-lo': [0,      0,   0,   0,   0,   0,   0,      0,   0,   0],
        'tag-action-hi': [1,      1,   1,   1,   1,   1,   1,      1,   1,   1],
    })
    assert dfs_close(pf.data, expected)
    assert np.all(
        pf.decision_points == np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    )
    assert np.all(
        pf.action_change  ==  np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    )


def test_per_tag_overrides(dummy_app_state: AppState):
    raw_obs = pd.DataFrame({
        'tag_1': [1, 2, 3],
        'tag_2': [1, 2, 3],
        'action': [0, 0, 0],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1],
    })

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.REFRESH,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    sc = StateConstructor(
        dummy_app_state,
        tag_cfgs=[
            BasicTagConfig(name='tag_1'),
            BasicTagConfig(
                name='tag_2',
                state_constructor=[
                    TraceConfig(trace_values=[0.1]),
                ],
            ),
            BasicTagConfig(name='action'),
        ],
        cfg=SCConfig(
            defaults=[
                TraceConfig(trace_values=[0.1, 0.01]),
                AddRawConfig(),
            ],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=4),
                obs_period=timedelta(minutes=1),
                normalize=False,
            ),
        ),
    )

    pf = sc(pf)
    expected = pd.DataFrame({
        'tag_1':             [1, 2, 3],
        'tag_1_trace-0.1':   [1., 1.9, 2.89],
        'tag_1_trace-0.01':  [1., 1.99, 2.9899],
        'tag_2_trace-0.1':   [1., 1.9, 2.89],
        'action':            [0, 0, 0],
        'action_trace-0.1':  [0, 0, 0],
        'action_trace-0.01': [0, 0, 0],
        'action-lo':         [0, 0, 0],
        'action-hi':         [1, 1, 1],
    })
    assert dfs_close(pf.data, expected)
