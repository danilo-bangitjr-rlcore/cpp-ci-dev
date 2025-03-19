import datetime
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from torch import tensor

from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.constructors.sc import SCConfig
from corerl.data_pipeline.datatypes import DataMode, Step, Transition
from corerl.data_pipeline.imputers.per_tag.copy import CopyImputerConfig
from corerl.data_pipeline.imputers.per_tag.linear import LinearImputerConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import IdentityConfig, NullConfig, ScaleConfig
from corerl.data_pipeline.transforms.comparator import ComparatorConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.data_pipeline.transition_filter import TransitionFilterConfig
from corerl.state import AppState
from test.infrastructure.utils.pandas import dfs_close


def test_pipeline1(dummy_app_state: AppState):
    cfg = PipelineConfig(
        tags=[
            TagConfig(
                name='tag-1',
                filter=[
                    ScaleConfig(factor=0.5),
                    ComparatorConfig(op='==', val=2),
                ],
                preprocess=[],
                state_constructor=[],
                is_endogenous=False
            ),
            TagConfig(
                name='tag-2',
                operating_range=(None, 10),
                red_bounds=(-1, None),
                imputer=LinearImputerConfig(max_gap=2),
                preprocess=[],
                state_constructor=[
                    NormalizerConfig(min=0, max=10),
                    TraceConfig(trace_values=[0.1]),
                ],
                is_endogenous=True
            ),
            TagConfig(
                name='action-1',
                operating_range=(0, 1),
                preprocess=[],
                action_constructor=[],
                state_constructor=[NullConfig()],
            ),
            TagConfig(
                name="reward",
                preprocess=[],
                state_constructor=[NullConfig()],
                reward_constructor=[IdentityConfig()],
            ),
        ],
        transition_creator=AllTheTimeTCConfig(
            max_n_step=2,
            gamma=0.9,
        ),
        transition_filter=TransitionFilterConfig(
            filters=[
                'only_no_action_change',
                'no_nan'
            ],
        ),
        state_constructor=SCConfig(
            countdown=CountdownConfig(
                action_period=timedelta(minutes=5),
                obs_period=timedelta(minutes=5),
                kind='int',
                normalize=False,
            ),
        ),
    )

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-1', 'tag-2', 'reward', 'action-1']
    df = pd.DataFrame(
        data=[
            # note alternation between actions
            [np.nan, 0,        0,    0],
            [0,      2,        3,    1],
            [1,      4,        0,    0],
            [2,      6,        0,    1],
            [np.nan, np.nan,   0,    0],
            [4,      10,       1,    1],
            # tag-2 is out-of-bounds
            [5,      12,       0,    0],
        ],
        columns=cols,
        index=idx,
    )

    pipeline = Pipeline(dummy_app_state, cfg)
    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # returned df has columns sorted in order: action, endogenous, exogenous, state, reward
    cols = ['tag-1', 'countdown.[0]', 'tag-2_norm_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 0,      0],
            [0,      0,      0.18],
            [1,      0,      0.378],
            [2,      0,      0.5778],
            [np.nan, 0,      0.77778],
            [np.nan, 0,      0.977778],
            [5,      0,      np.nan],
        ],
        columns=cols,
        index=idx,
    )

    expected_reward = pd.DataFrame(
        data=[
            [0],
            [3],
            [0],
            [0],
            [0],
            [1],
            [0],
        ],
        columns=['reward'],
        index=idx,
    )

    # breakpoint()
    assert dfs_close(got.df, expected_df, col_order_matters=True)
    assert dfs_close(got.rewards, expected_reward)
    assert got.transitions == [
        # notice that the first row of the DF was skipped due to the np.nan
        Transition(
            steps=[
                # expected state order: states sorted. Thus, [tag-1, countdown.[0], tag-2_norm_trace-0.1]
                Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([0.0, 0, 0.18]), dp=True, ac=True),
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 0, 0.378]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9
        ),
        Transition(
            steps=[
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 0,  0.378]), dp=True, ac=True),
                Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([2.0, 0,  0.5778]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9
        ),
    ]


def test_pipeline2(dummy_app_state: AppState):
    cfg = PipelineConfig(
        tags=[
            TagConfig(
                name='tag-1',
                imputer=CopyImputerConfig(imputation_horizon=2),
                preprocess=[],
                state_constructor=[],
                is_endogenous=False,
            ),
            TagConfig(
                name='tag-2',
                preprocess=[NormalizerConfig(min=0, max=12)],
                operating_range=(0, 12),
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    TraceConfig(trace_values=[0.1]),
                ],
                is_endogenous=True
            ),
            TagConfig(
                name='action-1',
                preprocess=[],
                operating_range=(0, 1),
                action_constructor=[],
                state_constructor=[],
            ),
            TagConfig(
                name="reward",
                preprocess=[],
                reward_constructor=[IdentityConfig()],
                state_constructor=[NullConfig()],
            ),
        ],
        transition_creator=AllTheTimeTCConfig(
            max_n_step=1,
            gamma=0.9,
        ),
        transition_filter=TransitionFilterConfig(
            filters=[
                'only_no_action_change',
                'only_post_dp',
                'no_nan'
            ],
        ),
        state_constructor=SCConfig(
            countdown=CountdownConfig(
                action_period=timedelta(minutes=5),
                obs_period=timedelta(minutes=5),
                kind='int',
                normalize=False,
            ),
        ),
    )

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-1', 'tag-2', 'reward', 'action-1']
    df = pd.DataFrame(
        data=[
            [np.nan, 0,        0,    0],
            [0,      2,        3,    1],
            [1,      4,        0,    0],
            [np.nan, 6,        0,    1],
            [np.nan, np.nan,   0,    0],
            # note tag-2 is out-of-bounds
            [4,      20,       1,    1],
            [np.nan, 12,       0,    0],
        ],
        columns=cols,
        index=idx,
    )

    pipeline = Pipeline(dummy_app_state, cfg)
    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # returned df has columns sorted in order: action, endogenous, exogenous, state
    cols = ['action-1', 'tag-1', 'countdown.[0]', 'tag-2_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [0,    0,     0,     0],
            [1,    0,     0,     0.15],
            [0,    1,     0,     0.315],
            [1,    1,     0,     0.4815],
            [0,    1,     0,     0.64815],
            [1,    4,     0,     0.814815],
            [0,    4,     0,     0.981482],
        ],
        columns=cols,
        index=idx,
    )

    assert dfs_close(got.df, expected_df, col_order_matters=True)
    assert got.transitions == [
        # notice that the first row of the DF was skipped due to the np.nan
        Transition(
            steps=[
                # countdown comes first in the state
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 0., 0, 0.0]), dp=True, ac=True),
                Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([1., 0., 0, 0.15]), dp=True, ac=True),
            ],
            n_step_reward=3.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([1., 0., 0, 0.15]), dp=True, ac=True),
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 0, 0.315]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 0, 0.315]), dp=True, ac=True),
                Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([1., 1., 0, 0.4815]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([1., 1., 0, 0.4815]), dp=True, ac=True),
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 0, 0.64815]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 0, 0.64815]), dp=True, ac=True),
                Step(reward=1, action=tensor([1.]), gamma=0.9, state=tensor([1., 4., 0, 0.814815]), dp=True, ac=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=1, action=tensor([1.]), gamma=0.9, state=tensor([1., 4., 0, 0.814815]), dp=True, ac=True),
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 4., 0, 0.981482]), dp=True, ac=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
    ]


def test_delta_action_pipeline(dummy_app_state: AppState):
    cfg = direct_load_config(
        PipelineConfig,
        base='test/small/data_pipeline/end_to_end/assets',
        config_name='delta_action.yaml',
    )
    assert isinstance(cfg, PipelineConfig)


    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-0', 'reward', 'action-0']
    df = pd.DataFrame(
        data=[
            [np.nan,  0,   -1],
            [0,       3,    1],
            [1,       0,    2],
            [2,       0,    4], # action is out-of-bounds
            [np.nan,  0,    2],
            [4,       1,    1],
            [5,       0,   -2],
        ],
        columns=cols,
        index=idx,
    )

    pipeline = Pipeline(dummy_app_state, cfg)
    got = pipeline(df, data_mode=DataMode.ONLINE)

    # NOTE: the action-0 column are normalized actions
    # as a result of the preprocess stage.
    # Therefore, the delta actions are in normalized space,
    # and are then normalized _again_ to be between [0, 1].
    # So a normalized delta of 1.0 is a delta of 0.5 in
    # the normed action space, or a delta of 2 in the original
    # action space.
    expected_actions = pd.DataFrame(
        data=[
            [0.25,  np.nan],
            [0.75,  1.0],
            [1.,    0.75],
            [1.,    0.5],
            [1.,    0.5],
            [0.75,  0.25],
            [0.,    np.nan],
        ],
        columns=['action-0', 'action-0_Δ_bounds_norm'],
        index=idx,
    )

    pd.testing.assert_frame_equal(got.actions, expected_actions)

def test_delta_action_countdown(dummy_app_state: AppState):
    """
    This is a regression test for a bug in which a change in delta triggered
    a reset of the countdown feature even when the direct action remained the same
    """
    cfg = direct_load_config(
        PipelineConfig,
        base='test/small/data_pipeline/end_to_end/assets',
        config_name='delta_action.yaml',
    )
    assert isinstance(cfg, PipelineConfig)

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(9)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-0', 'reward', 'action-0']
    df = pd.DataFrame(
        data=[
            [np.nan,  0,   -1],
            [np.nan,  0,   -1],
            [np.nan,  0,   -1],
            [np.nan,  0,   -1],
            [np.nan,  0,   -1],
            [1,       0,    2], # delta nonzero
            [2,       0,    2], # delta zero
            [np.nan,  0,    2],
            [np.nan,  0,    2],
        ],
        columns=cols,
        index=idx,
    )

    pipeline = Pipeline(dummy_app_state, cfg)
    got = pipeline(df, data_mode=DataMode.ONLINE)

    # make sure change in delta did not trigger cd reset
    cd = np.array(got.states['countdown.[0]'].values)
    expected = np.array([0, 3, 2, 1, 0, 3, 2, 1, 0])
    # wrong = np.array([0, 3, 2, 1, 0, 3, 3, 2, 1])

    assert np.allclose(cd, expected)
