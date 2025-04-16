import datetime
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from corerl.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.constructors.sc import SCConfig
from corerl.data_pipeline.datatypes import DataMode, StageCode
from corerl.data_pipeline.imputers.per_tag.linear import LinearImputerConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig, TagType
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.state import AppState
from test.infrastructure.utils.pandas import dfs_close


def test_construct_pipeline(dummy_app_state: AppState):
    cfg = PipelineConfig(
        tags=[
            TagConfig(name='sensor_x', operating_range=(-1, 1)),
            TagConfig(name='sensor_y', red_bounds=(1.1, 3.3)),
        ],
        transition_creator=AllTheTimeTCConfig(
            # set arbitrarily
            gamma=0.9,
            min_n_step=1,
            max_n_step=30
        ),
        state_constructor=SCConfig(
            countdown=CountdownConfig(
                action_period=timedelta(minutes=15),
                obs_period=timedelta(minutes=15),
            ),
        ),
    )
    _ = Pipeline(dummy_app_state, cfg)


def test_passing_data_to_pipeline(dummy_app_state: AppState):
    cfg = PipelineConfig(
        tags=[
            TagConfig(preprocess=[NormalizerConfig(from_data=True)], name='sensor_x', operating_range=(-3, 3)),
            TagConfig(preprocess=[NormalizerConfig(from_data=True)], name='sensor_y', red_bounds=(1.1, 3.3)),
        ],
        transition_creator=AllTheTimeTCConfig(
            # set arbitrarily
            gamma=0.9,
            min_n_step=1,
            max_n_step=30
        ),
        state_constructor=SCConfig(
            defaults=[NormalizerConfig(from_data=True)],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=15),
                obs_period=timedelta(minutes=15),
            ),
        ),
    )
    pipeline = Pipeline(dummy_app_state, cfg)

    cols = {"sensor_x": [np.nan, 1.0, 2.0], "sensor_y": [2.0, np.nan, 3.0], "reward": [1., 2., 3.]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, 1),
        datetime.datetime(2024, 1, 1, 1, 2),
        datetime.datetime(2024, 1, 1, 1, 3),
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)

    # test that we can run the pf through the pipeline
    _ = pipeline(df, data_mode=DataMode.OFFLINE)


def test_state_action_dim(dummy_app_state: AppState):
    cfg = PipelineConfig(
        tags=[
            TagConfig(preprocess=[NormalizerConfig(from_data=True)], name='tag-1'),
            TagConfig(
                name='tag-2',
                operating_range=(None, 10),
                yellow_bounds=(-1, None),
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    NormalizerConfig(from_data=True),
                    TraceConfig(trace_values=[0.1, 0.9]),
                ],
            ),
            TagConfig(name='tag-3', operating_range=(0, 1), type=TagType.ai_setpoint),
            TagConfig(name='tag-4', operating_range=(0, 1), type=TagType.ai_setpoint),
        ],
        state_constructor=SCConfig(
            countdown=CountdownConfig(
                action_period=timedelta(minutes=5),
                obs_period=timedelta(minutes=5),
            ),
            defaults=[],
        ),
        transition_creator=AllTheTimeTCConfig(
            # set arbitrarily
            gamma=0.9,
            min_n_step=1,
            max_n_step=30
        ),
    )

    pipeline = Pipeline(dummy_app_state, cfg)

    col_desc = pipeline.column_descriptions
    assert col_desc.state_dim == 5
    assert col_desc.action_dim == 2


def test_sub_pipeline1(dummy_app_state: AppState):
    cfg = PipelineConfig(
        tags=[
            TagConfig(
                name='tag-1',
                preprocess=[NormalizerConfig(min=0, max=5)],
                state_constructor=[],
            ),
            TagConfig(
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
            max_n_step=30
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
