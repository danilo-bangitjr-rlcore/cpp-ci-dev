import datetime
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from torch import tensor

from corerl.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.constructors.sc import SCConfig
from corerl.data_pipeline.datatypes import CallerCode, Step, Transition
from corerl.data_pipeline.imputers.copy import CopyImputerConfig
from corerl.data_pipeline.imputers.linear import LinearImputerConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import NullConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.data_pipeline.transition_filter import TransitionFilterConfig
from test.infrastructure.utils.pandas import dfs_close


def test_pipeline1():
    cfg = PipelineConfig(
        tags=[
            TagConfig(
                name='tag-1',
                state_constructor=[],
                is_endogenous=False
            ),
            TagConfig(
                name='tag-2',
                bounds=(None, 10),
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    NormalizerConfig(),
                    TraceConfig(trace_values=[0.1]),
                ],
            ),
            TagConfig(name='action-1', action_constructor=[], state_constructor=[NullConfig()]),
            TagConfig(name="reward", is_meta=True),
        ],
        transition_creator=AllTheTimeTCConfig(
            max_n_step=2,
            gamma=0.9,
        ),
        transition_filter=TransitionFilterConfig(
            filters=[
                'only_no_action_change',
            ],
        ),
        state_constructor=SCConfig(
            countdown=CountdownConfig(
                action_period=timedelta(minutes=5),
                obs_period=timedelta(minutes=5),
                kind='int'
            ),
        ),
        obs_period=timedelta(minutes=5),
        action_period=timedelta(minutes=5),
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

    pipeline = Pipeline(cfg)
    got = pipeline(
        df,
        caller_code=CallerCode.ONLINE,
    )

    # returned df has columns sorted in order: action, endogenous, exogenous, state, reward
    cols = ['tag-1', 'countdown.[0]', 'tag-2_norm_trace-0.1', 'reward']
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 1,      0,         0],
            [0,      1,      0.18,      3],
            [1,      1,      0.378,     0],
            [2,      1,      0.5778,    0],
            [np.nan, 1,      0.77778,   0],
            [4,      1,      0.977778,  1],
            [5,      1,      np.nan,    0],
        ],
        columns=cols,
        index=idx,
    )

    assert dfs_close(got.df, expected_df, col_order_matters=True)
    assert got.transitions == [
        # notice that the first row of the DF was skipped due to the np.nan
        Transition(
            steps=[
                # countdown is first in the state
                Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([0., 1, 0.18]), dp=True),
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 1, 0.378]), dp=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9
        ),
        Transition(
            steps=[
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 1,  0.378]), dp=True),
                Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([2.0, 1,  0.5778]), dp=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9
        ),
    ]


def test_pipeline2():
    cfg = PipelineConfig(
        tags=[
            TagConfig(
                name='tag-1',
                imputer=CopyImputerConfig(imputation_horizon=2),
                state_constructor=[],
            ),
            TagConfig(
                name='tag-2',
                bounds=(None, 12),
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    NormalizerConfig(),
                    TraceConfig(trace_values=[0.1]),
                ],
            ),
            TagConfig(name='action-1', action_constructor=[], state_constructor=[]),
            TagConfig(name="reward", is_meta=True),
        ],
        transition_creator=AllTheTimeTCConfig(
            max_n_step=1,
            gamma=0.9,
        ),
        transition_filter=TransitionFilterConfig(
            filters=[
                'only_no_action_change',
                'only_post_dp',
            ],
        ),
        state_constructor=SCConfig(
            countdown=CountdownConfig(
                action_period=timedelta(minutes=5),
                obs_period=timedelta(minutes=5),
                kind='int',
            ),
        ),
        obs_period=timedelta(minutes=5),
        action_period=timedelta(minutes=5),
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

    pipeline = Pipeline(cfg)
    got = pipeline(
        df,
        caller_code=CallerCode.ONLINE,
    )

    # returned df has columns sorted in order: action, endogenous, exogenous, state, reward
    cols = ['action-1', 'tag-1', 'countdown.[0]', 'tag-2_norm_trace-0.1', 'reward']
    expected_df = pd.DataFrame(
        data=[
            [0,    0,     1,     0,              0],
            [1,    0,     1,     0.15,           3],
            [0,    1,     1,     0.315,          0],
            [1,    1,     1,     0.4815,         0],
            [0,    1,     1,     0.64815,        0],
            [1,    4,     1,     0.814815,       1],
            [0,    4,     1,     0.981482,       0],
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
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 0., 1, 0.0]), dp=True),
                Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([1., 0., 1, 0.15]), dp=True),
            ],
            n_step_reward=3.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([1., 0., 1, 0.15]), dp=True),
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 1, 0.315]), dp=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 1, 0.315]), dp=True),
                Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([1., 1., 1, 0.4815]), dp=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([1., 1., 1, 0.4815]), dp=True),
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 1, 0.64815]), dp=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 1, 0.64815]), dp=True),
                Step(reward=1, action=tensor([1.]), gamma=0.9, state=tensor([1., 4., 1, 0.814815]), dp=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(reward=1, action=tensor([1.]), gamma=0.9, state=tensor([1., 4., 1, 0.814815]), dp=True),
                Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 4., 1, 0.981482]), dp=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
    ]
