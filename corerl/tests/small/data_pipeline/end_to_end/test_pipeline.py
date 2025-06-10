import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from test.infrastructure.utils.pandas import dfs_close

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.datatypes import DataMode, Step, Transition
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState


def mkstep(
    reward: float,
    action: jax.Array,
    gamma: float,
    state: jax.Array,
    dp: bool, # decision point
    ac: bool, # action change
):
    return Step(
        reward=reward,
        action=action,
        gamma=gamma,
        state=state,
        action_lo=jnp.zeros_like(action),
        action_hi=jnp.ones_like(action),
        dp=dp,
        ac=ac,
    )

def test_pipeline1():
    cfg = direct_load_config(MainConfig, config_name='tests/small/data_pipeline/end_to_end/test_pipeline1.yaml')
    assert isinstance(cfg, MainConfig)

    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=DummyEventBus(),
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

    pipeline = Pipeline(app_state, cfg.pipeline)
    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # returned df has columns sorted in order: action, endogenous, exogenous, state, reward
    cols = ['tag-1', 'action-1-hi', 'action-1-lo', 'countdown.[0]', 'tag-2_norm_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 1, 0, 0, 0],
            [0,      1, 0, 0, 0.18],
            [1,      1, 0, 0, 0.378],
            [2,      1, 0, 0, 0.5778],
            [np.nan, 1, 0, 0, 0.77778],
            [np.nan, 1, 0, 0, 0.977778],
            [5,      1, 0, 0, np.nan],
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
                # expected state order: [tag-1, action-1-hi, action-1-lo, countdown.[0], tag-2_norm_trace-0.1]
                mkstep(reward=3,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([0.0, 1, 0, 0, 0.18]),
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([1.0, 1, 0, 0, 0.378]),
                       dp=True,
                       ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([1.0, 1, 0, 0, 0.378]),
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([2.0, 1, 0, 0,  0.5778]),
                       dp=True,
                       ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
    ]


def test_pipeline2():
    cfg = direct_load_config(MainConfig, config_name='tests/small/data_pipeline/end_to_end/test_pipeline2.yaml')
    assert isinstance(cfg, MainConfig)

    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=DummyEventBus(),
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

    pipeline = Pipeline(app_state, cfg.pipeline)
    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # returned df has columns sorted in order: action, endogenous, exogenous, state
    cols = ['action-1', 'tag-1', 'action-1-hi', 'action-1-lo', 'countdown.[0]', 'tag-2_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0.15],
            [0, 1, 1, 0, 0, 0.315],
            [1, 1, 1, 0, 0, 0.4815],
            [0, 1, 1, 0, 0, 0.64815],
            [1, 4, 1, 0, 0, 0.814815],
            [0, 4, 1, 0, 0, 0.981482],
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
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0., 0., 1, 0, 0, 0.0]),
                       dp=True,
                       ac=True),
                mkstep(reward=3,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1., 0., 1, 0, 0, 0.15]),
                       dp=True,
                       ac=True),
            ],
            n_step_reward=3.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=3,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1., 0., 1, 0, 0, 0.15]),
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0., 1., 1, 0, 0, 0.315]),
                       dp=True,
                       ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0., 1., 1, 0, 0, 0.315]),
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1., 1., 1, 0, 0, 0.4815]),
                       dp=True,
                       ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1., 1., 1, 0, 0, 0.4815]),
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0., 1., 1, 0, 0, 0.64815]),
                       dp=True,
                       ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0., 1., 1, 0, 0, 0.64815]),
                       dp=True,
                       ac=True),
                mkstep(reward=1,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1., 4., 1, 0, 0, 0.814815]),
                       dp=True,
                       ac=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=1,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1., 4., 1, 0, 0, 0.814815]),
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0., 4., 1, 0, 0, 0.981482]),
                       dp=True,
                       ac=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
    ]

def test_pipeline3():
    cfg = direct_load_config(MainConfig, config_name='tests/small/data_pipeline/end_to_end/test_pipeline3.yaml')
    assert isinstance(cfg, MainConfig)

    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=DummyEventBus(),
    )

    start = datetime.datetime(2024, 7, 13, 10, tzinfo=datetime.UTC)
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

    pipeline = Pipeline(app_state, cfg.pipeline)
    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
    )

    # returned df has columns sorted in order: action, endogenous, exogenous, state
    cols = ['action-1', 'tag-1', 'action-1-hi', 'action-1-lo', 'countdown.[0]', 'day_of_week_0', 'day_of_week_1',
            'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'tag-2_trace-0.1',
            'time_of_day_cos', 'time_of_day_sin', 'time_of_year_cos', 'time_of_year_sin']
    expected_df = pd.DataFrame(
        data=[
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -0.866025, 0.5, -0.978856, -0.204552],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.15, -0.876727, 0.480989, -0.978856, -0.204552],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.315, -0.887011, 0.461749, -0.978856, -0.204552],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.4815, -0.896873, 0.442289, -0.978856, -0.204552],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.64815, -0.906308, 0.422618, -0.978856, -0.204552],
            [1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.814815, -0.915311, 0.402747, -0.978856, -0.204552],
            [0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.981482, -0.923880, 0.382683, -0.978856, -0.204552],
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
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -0.866025, 0.5, -0.978856, -0.204552]),
                       dp=True,
                       ac=True),
                mkstep(reward=3,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.15, -0.876727, 0.480989, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
            ],
            n_step_reward=3.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=3,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.15, -0.876727, 0.480989, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.315, -0.887011, 0.461749, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.315, -0.887011, 0.461749, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.4815, -0.896873, 0.442289, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.4815, -0.896873, 0.442289, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.64815, -0.906308, 0.422618, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.64815, -0.906308, 0.422618, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
                mkstep(reward=1,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.814815, -0.915311, 0.402747, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                mkstep(reward=1,
                       action=jnp.array([1.]),
                       gamma=0.9,
                       state=jnp.array([1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.814815, -0.915311, 0.402747, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
                mkstep(reward=0,
                       action=jnp.array([0.]),
                       gamma=0.9,
                       state=jnp.array([0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.981482, -0.923880, 0.382683, -0.978856, -0.204552]), # noqa: E501
                       dp=True,
                       ac=True),
            ],
            n_step_reward=1.,
            n_step_gamma=0.9,
        ),
    ]
