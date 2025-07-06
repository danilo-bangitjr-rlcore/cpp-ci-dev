import datetime
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config
from test.infrastructure.utils.pandas import dfs_close

from corerl.config import MainConfig
from corerl.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.constructors.sc import SCConfig
from corerl.data_pipeline.datatypes import DataMode, StageCode
from corerl.data_pipeline.imputers.per_tag.linear import LinearImputerConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.state import AppState
from corerl.tags.setpoint import SetpointTagConfig
from corerl.tags.tag_config import BasicTagConfig


@pytest.fixture
def pipeline1_config():
    cfg = direct_load_config(MainConfig, config_name='tests/small/data_pipeline/end_to_end/test_pipeline1.yaml')
    assert not isinstance(cfg, ConfigValidationErrors)
    return cfg


def test_construct_pipeline(dummy_app_state: AppState, pipeline1_config: MainConfig):
    _ = Pipeline(dummy_app_state, pipeline1_config.pipeline)


def test_passing_data_to_pipeline(dummy_app_state: AppState, pipeline1_config: MainConfig):
    pipeline = Pipeline(dummy_app_state, pipeline1_config.pipeline)

    cols = {
        "tag-1": [np.nan, 0, 1],
        "tag-2": [0, 2, 4],
        "reward": [1., 2., 3.],
        "action-1": [0, 1, 0],
    }
    dates = [
        datetime.datetime(2024, 1, 1, 1, 1),
        datetime.datetime(2024, 1, 1, 1, 6),
        datetime.datetime(2024, 1, 1, 1, 11),
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)

    # test that we can run the pf through the pipeline
    _ = pipeline(df, data_mode=DataMode.OFFLINE)


def test_state_action_dim(dummy_app_state: AppState, pipeline1_config: MainConfig):
    pipeline = Pipeline(dummy_app_state, pipeline1_config.pipeline)
    col_desc = pipeline.column_descriptions
    assert col_desc.state_dim == 5
    assert col_desc.action_dim == 1
        ),
        transition_creator=AllTheTimeTCConfig(
            # set arbitrarily
            gamma=0.9,
            min_n_step=1,
            max_n_step=30,
        ),
    )

    pipeline = Pipeline(dummy_app_state, cfg)

    col_desc = pipeline.column_descriptions
    assert col_desc.state_dim == 9
    assert col_desc.action_dim == 2


def test_sub_pipeline1(dummy_app_state: AppState):
    cfg = PipelineConfig(
        tags=[
            BasicTagConfig(
                name='tag-1',
                preprocess=[NormalizerConfig(min=0, max=5)],
                state_constructor=[],
            ),
            BasicTagConfig(
                name='tag-2',
                operating_range=(None, 10),
                preprocess=[NormalizerConfig(min=0, max=10)],
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    TraceConfig(trace_values=[0.1]),
                ],
            ),
        ],
        transition_creator=AllTheTimeTCConfig(
            # set arbitrarily
            gamma=0.9,
            min_n_step=1,
            max_n_step=30,
        ),
        state_constructor=SCConfig(
            defaults=[NormalizerConfig(from_data=True)],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=5),
                obs_period=timedelta(minutes=5),
            ),
        ),
    )

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-1', 'tag-2']
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0,      2],
            [1,      4],
            [2,      6],
            [np.nan, np.nan],
            [4,      10],
            [5,      12],
        ],
        columns=cols,
        index=idx,
    )

    pipeline = Pipeline(dummy_app_state, cfg)
    got = pipeline(
        df,
        data_mode=DataMode.ONLINE,
        stages=(
            StageCode.PREPROCESS,
            StageCode.BOUNDS,
            StageCode.ODDITY,
            StageCode.IMPUTER,
            StageCode.AC,
            StageCode.SC,
        ),
    )

    cols = ['tag-1', 'tag-2_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [np.nan,   0],
            [0,        0.18],
            [0.2,      0.378],
            [0.4,      0.5778],
            [np.nan,   0.77778],
            [0.8,      0.977778],
            [1.0,      np.nan],
        ],
        columns=cols,
        index=idx,
    )

    assert dfs_close(got.df, expected_df)
    assert got.transitions is None


def test_pipeline_overlapping_time(dummy_app_state: AppState):
    cfg = direct_load_config(MainConfig, config_name='tests/small/data_pipeline/end_to_end/test_pipeline1.yaml')
    assert not isinstance(cfg, ConfigValidationErrors)
    pipeline = Pipeline(dummy_app_state, cfg.pipeline)

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
            [2,      6,        0,    1],
            [np.nan, np.nan,   0,    0],
            [4,      10,       1,    1],
            [5,      12,       0,    0],
        ],
        columns=cols,
        index=idx,
    )

    pipeline(df)

    # NOTE: overlap time by two time steps
    dates = [start + (i - 2) * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

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
        columns=cols,
        index=idx,
    )

    out = pipeline(prior_df)

    # this can only be true if the temporal state is being reset
    # between invocations
    assert np.isclose(
        out.states['tag-2_norm_trace-0.1'].iloc[0],
        0.1 * first_value,
    )
