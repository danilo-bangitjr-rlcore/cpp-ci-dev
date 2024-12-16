from typing import Any
import numpy as np
import pandas as pd
import datetime
from torch import tensor

from corerl.data_pipeline.imputers.linear import LinearImputerConfig
from corerl.data_pipeline.imputers.copy import CopyImputerConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.state_constructors.sc import SCConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import Step, CallerCode, NewTransition
from corerl.data_pipeline.transition_creators.anytime import AnytimeTransitionCreatorConfig
from test.infrastructure.utils.pandas import dfs_close

def test_pipeline1():
    cfg = PipelineConfig(
        tags=[
            TagConfig(
                name='tag-1',
                state_constructor=[],
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
            TagConfig(name='action-1', is_action=True),
        ],
        agent_transition_creator=AnytimeTransitionCreatorConfig(
            steps_per_decision=2,
            gamma=0.9,
            n_step=None,
            only_dp_transitions=False,
        ),
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
        obs_interval_minutes=5,
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

    cols = ['tag-1', 'tag-2_norm_trace-0.1', 'reward', 'action-1']
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 0,         0,    0],
            [0,      0.18,      3,    1],
            [1,      0.378,     0,    0],
            [2,      0.5778,    0,    1],
            [np.nan, 0.77778,   0,    0],
            [4,      0.977778,  1,    1],
            [5,      np.nan,    0,    0],
        ],
        columns=cols,
        index=idx,
    )

    assert dfs_close(got.df, expected_df)
    assert got.transitions == [
        # notice that the first row of the DF was skipped due to the np.nan
        NewTransition(
            prior=Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([0., 0.18, 0, 1]), dp=True),
            post=Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 0.378, 1, 0]), dp=False),
            n_steps=1,
        ),
        NewTransition(
            prior=Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 0.378, 0, 1]), dp=True),
            post=Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([2.0, 0.5778, 1, 0]), dp=False),
            n_steps=1,
        )
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
            TagConfig(name='action-1', is_action=True),
        ],
        agent_transition_creator=AnytimeTransitionCreatorConfig(
            steps_per_decision=2,
            gamma=0.9,
            n_step=None,
            only_dp_transitions=False,
        ),
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
        obs_interval_minutes=5,
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

    cols = ['tag-1', 'tag-2_norm_trace-0.1', 'reward', 'action-1']
    expected_df = pd.DataFrame(
        data=[
            [0,      0,         0,    0],
            [0,      0.15,      3,    1],
            [1,      0.315,     0,    0],
            [1,      0.4815,    0,    1],
            [1,      0.64815,   0,    0],
            [4,      0.814815,  1,    1],
            [4,      0.981482,  0,    0],
        ],
        columns=cols,
        index=idx,
    )

    assert dfs_close(got.df, expected_df)
    assert got.transitions == [
        # notice that the first row of the DF was skipped due to the np.nan
        NewTransition(
            prior=Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 0.0, 0, 1]), dp=True),
            post=Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([0., 0.15, 1, 0]), dp=False),
            n_steps=1,
        ),
        NewTransition(
            prior=Step(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([0., 0.15, 0, 1]), dp=True),
            post=Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1., 0.315, 1, 0]), dp=False),
            n_steps=1,
        ),
        NewTransition(
            prior=Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1., 0.315, 0, 1]), dp=True),
            post=Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([1., 0.4815, 1, 0]), dp=False),
            n_steps=1,
        ),
        NewTransition(
            prior=Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([1., 0.4815, 0, 1]), dp=True),
            post=Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1., 0.64815, 1, 0]), dp=False),
            n_steps=1,
        ),
        NewTransition(
            prior=Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1., 0.64815, 0, 1]), dp=True),
            post=Step(reward=1, action=tensor([1.]), gamma=0.9, state=tensor([4., 0.814815, 1, 0]), dp=False),
            n_steps=1,
        ),
    ]
