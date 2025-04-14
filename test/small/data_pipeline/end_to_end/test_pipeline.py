import datetime
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, tensor

from corerl.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.constructors.sc import SCConfig
from corerl.data_pipeline.datatypes import DataMode, Step, Transition
from corerl.data_pipeline.imputers.per_tag.copy import CopyImputerConfig
from corerl.data_pipeline.imputers.per_tag.linear import LinearImputerConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig, TagType
from corerl.data_pipeline.transforms import IdentityConfig, NullConfig, ScaleConfig
from corerl.data_pipeline.transforms.comparator import ComparatorConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.data_pipeline.transition_filter import TransitionFilterConfig
from corerl.state import AppState
from test.infrastructure.utils.pandas import dfs_close


def mkstep(
    reward: float,
    action: Tensor,
    gamma: float,
    state: Tensor,
    dp: bool, # decision point
    ac: bool, # action change
):
    return Step(
        reward=reward,
        action=action,
        gamma=gamma,
        state=state,
        action_lo=torch.zeros_like(action),
        action_hi=torch.ones_like(action),
        dp=dp,
        ac=ac
    )

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
                type=TagType.ai_setpoint,
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
                mkstep(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([0.0, 0, 0.18]), dp=True, ac=True),
                mkstep(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 0, 0.378]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9
        ),
        Transition(
            steps=[
                mkstep(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 0,  0.378]), dp=True, ac=True),
                mkstep(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([2.0, 0,  0.5778]), dp=True, ac=True),
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
                type=TagType.ai_setpoint,
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
                mkstep(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 0., 0, 0.0]), dp=True, ac=True),
                mkstep(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([1., 0., 0, 0.15]), dp=True, ac=True),
            ],
            n_step_reward=3.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=3, action=tensor([1.]), gamma=0.9, state=tensor([1., 0., 0, 0.15]), dp=True, ac=True),
                mkstep(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 0, 0.315]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 0, 0.315]), dp=True, ac=True),
                mkstep(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([1., 1., 0, 0.4815]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([1., 1., 0, 0.4815]), dp=True, ac=True),
                mkstep(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 0, 0.64815]), dp=True, ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 1., 0, 0.64815]), dp=True, ac=True),
                mkstep(reward=1, action=tensor([1.]), gamma=0.9, state=tensor([1., 4., 0, 0.814815]), dp=True, ac=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=1, action=tensor([1.]), gamma=0.9, state=tensor([1., 4., 0, 0.814815]), dp=True, ac=True),
                mkstep(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([0., 4., 0, 0.981482]), dp=True, ac=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
    ]
