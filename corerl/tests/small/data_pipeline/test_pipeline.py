import jax.numpy as jnp
import numpy as np
import pandas as pd
import pandas.testing as pdt
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config
from test.infrastructure.app_state import make_dummy_app_state

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode, Step, Transition
from corerl.data_pipeline.pipeline import Pipeline
from corerl.state import AppState


def test_construct_pipeline(
    dummy_app_state: AppState,
    basic_config: MainConfig,
):
    _ = Pipeline(dummy_app_state, basic_config.pipeline)


def test_passing_data_to_pipeline(
    fake_pipeline_data: pd.DataFrame,
    dummy_pipeline: Pipeline,
):
    df = fake_pipeline_data
    _ = dummy_pipeline(df, data_mode=DataMode.OFFLINE)


def test_state_action_dim(
    dummy_app_state: AppState,
    basic_config: MainConfig,
):
    pipeline = Pipeline(dummy_app_state, basic_config.pipeline)
    col_desc = pipeline.column_descriptions
    assert col_desc.state_dim == 4
    assert col_desc.action_dim == 1


def test_pipeline1(
    basic_config: MainConfig,
    fake_pipeline_data: pd.DataFrame,
):
    app_state = make_dummy_app_state(basic_config)

    df = fake_pipeline_data
    idx = df.index
    pipeline = Pipeline(app_state, basic_config.pipeline)
    got = pipeline(df, data_mode=DataMode.ONLINE)

    cols = ['tag-1', 'action-1-hi', 'action-1-lo', 'tag-2_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 1,      0, 0],
            [0,      1,      0, 0.18],
            [1,      1,      0, 0.378],
            [2,      1,      0, 0.57780004],
            [np.nan, 1,      0, 0.57780004],
            [4,      1,      0, 0.57780004],
            [5,      1,      0, 0.57780004],
        ],
        columns=cols,
        index=idx,
    )

    expected_reward = pd.DataFrame(
        data=[
            [0.],
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

    pdt.assert_frame_equal(got.df, expected_df, check_exact=False, check_dtype=False, rtol=1e-5, atol=1e-8)
    pdt.assert_frame_equal(got.rewards, expected_reward, check_exact=False, check_dtype=False, rtol=1e-5, atol=1e-8)
    assert got.transitions == [
        Transition(
            steps=[
                Step(
                    reward=3,
                    action=jnp.array([1.]),
                    gamma=0.9,
                    state=jnp.array([0.0, 1, 0, 0.18]),
                    action_lo=jnp.zeros_like(jnp.array([1.])),
                    action_hi=jnp.ones_like(jnp.array([1.])),
                    dp=True,
                    ac=True,
                ),
                Step(
                    reward=0,
                    action=jnp.array([0.]),
                    gamma=0.9,
                    state=jnp.array([1.0, 1, 0, 0.378]),
                    action_lo=jnp.zeros_like(jnp.array([0.])),
                    action_hi=jnp.ones_like(jnp.array([0.])),
                    dp=True,
                    ac=True,
                ),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(
                    reward=0,
                    action=jnp.array([0.]),
                    gamma=0.9,
                    state=jnp.array([1.0, 1, 0, 0.378]),
                    action_lo=jnp.zeros_like(jnp.array([0.])),
                    action_hi=jnp.ones_like(jnp.array([0.])),
                    dp=True,
                    ac=True,
                ),
                Step(
                    reward=0,
                    action=jnp.array([1.]),
                    gamma=0.9,
                    state=jnp.array([2.0, 1, 0, 0.5778]),
                    action_lo=jnp.zeros_like(jnp.array([1.])),
                    action_hi=jnp.ones_like(jnp.array([1.])),
                    dp=True,
                    ac=True,
                ),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
        Transition(
            steps=[
                Step(
                    reward=1,
                    action=jnp.array([1.]),
                    gamma=0.9,
                    state=jnp.array([4.0, 1, 0, 0.5778]),
                    action_lo=jnp.zeros_like(jnp.array([0.])),
                    action_hi=jnp.ones_like(jnp.array([0.])),
                    dp=True,
                    ac=True,
                ),
                Step(
                    reward=0,
                    action=jnp.array([0.]),
                    gamma=0.9,
                    state=jnp.array([5.0, 1, 0, 0.5778]),
                    action_lo=jnp.zeros_like(jnp.array([1.])),
                    action_hi=jnp.ones_like(jnp.array([1.])),
                    dp=True,
                    ac=True,
                ),
            ],
            n_step_reward=0.,
            n_step_gamma=0.9,
        ),
    ]


def test_pipeline_overlapping_time(
    dummy_app_state: AppState,
    basic_config: MainConfig,
    fake_pipeline_data: pd.DataFrame,
):
    pipeline = Pipeline(dummy_app_state, basic_config.pipeline)
    df = fake_pipeline_data
    idx = df.index
    pipeline(df)

    # Overlap time by two time steps
    Δ = idx[1] - idx[0]
    shifted_idx = pd.DatetimeIndex([t - 2 * Δ for t in idx])
    first_value = 3
    prior_df = pd.DataFrame(
        data=[
            [3,  first_value,  0,    0],
            [0,      2,        3,    1],
            [1,      4,        0,    0],
            [2,      6,        0,    1],
            [np.nan, np.nan,   0,    0],
            [4,      10,       1,    1],
            [5,      np.nan,   np.nan, np.nan],
        ],
        columns=['tag-1', 'tag-2', 'reward', 'action-1'],
        index=shifted_idx,
    )
    out = pipeline(prior_df)

    # this can only be true if the temporal state is being reset
    # between invocations
    assert np.isclose(
        out.states['tag-2_trace-0.1'].iloc[0],
        0.1 * first_value,
    )


def test_duplicate_tag_names_raises():
    cfg_or_err = direct_load_config(MainConfig, config_name='tests/small/data_pipeline/assets/pipeline_dup_tag.yaml')
    assert isinstance(cfg_or_err, ConfigValidationErrors)

    # We need to make sure this error is specifically due to duplicate tag names
    # and not some other validation error.
    # However, we don't want to tightly couple this test to the exact error message.
    # In the future, we will have dedicated error types for different validation issues.
    assert "duplicate" in cfg_or_err.meta['pipeline'].message.lower()
