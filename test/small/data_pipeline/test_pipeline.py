from typing import Any
import numpy as np
import pandas as pd
import datetime

from corerl.data_pipeline.imputers.linear import LinearImputerConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.components.norm import NormalizerConfig
from corerl.data_pipeline.state_constructors.components.trace import TraceConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import CallerCode, StageCode
from corerl.data_pipeline.transition_creators.dummy import DummyTransitionCreatorConfig
from test.infrastructure.utils.pandas import dfs_close


def test_construct_pipeline():
    cfg = PipelineConfig(
        tags=[
            TagConfig(name='sensor_x'),
            TagConfig(name='sensor_y'),
        ],
        obs_interval_minutes=15,
        agent_transition_creator=DummyTransitionCreatorConfig(),
    )
    _ = Pipeline(cfg)


def test_passing_data_to_pipeline():
    cfg = PipelineConfig(
        tags=[
            TagConfig(name='sensor_x'),
            TagConfig(name='sensor_y'),
        ],
        obs_interval_minutes=15,
        agent_transition_creator=DummyTransitionCreatorConfig(),
    )
    pipeline = Pipeline(cfg)

    cols = {"sensor_x": [np.nan, 1.0], "sensor_y": [2.0, np.nan]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, 1 ),
        datetime.datetime(2024, 1, 1, 1, 2)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)

    # test that we can run the pf through the pipeline
    _ = pipeline(df, caller_code=CallerCode.OFFLINE)


def test_sub_pipeline1():
    cfg = PipelineConfig(
        tags=[
            TagConfig(name='tag-1'),
            TagConfig(
                name='tag-2',
                bounds=(None, 10),
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    NormalizerConfig(),
                    TraceConfig(trace_values=[0.1]),
                ],
            ),
        ],
        obs_interval_minutes=5,
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

    pipeline = Pipeline(cfg)
    got = pipeline(
        df,
        caller_code=CallerCode.ONLINE,
        stages=(StageCode.BOUNDS, StageCode.ODDITY, StageCode.IMPUTER, StageCode.SC),
    )

    cols: Any = ['tag-1', 'tag-2_norm_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0,      0.18],
            [1,      0.378],
            [2,      0.5778],
            [np.nan, 0.77778],
            [4,      0.977778],
            [5,      np.nan],
        ],
        columns=cols,
        index=idx,
    )

    assert dfs_close(got.df, expected_df)
    assert got.transitions is None


def test_sub_pipeline2():
    cfg = PipelineConfig(
        tags=[
            TagConfig(name='tag-1'),
            TagConfig(
                name='tag-2',
                bounds=(1, 2),
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    NormalizerConfig(),
                    TraceConfig(trace_values=[0.1]),
                ],
            ),
        ],
        obs_interval_minutes=5,
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

    pipeline = Pipeline(cfg)
    got = pipeline(
        df,
        caller_code=CallerCode.ONLINE,
        stages=(StageCode.IMPUTER, StageCode.SC),
    )

    cols: Any = ['tag-1', 'tag-2_norm_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0,      0.15],
            [1,      0.315],
            [2,      0.4815],
            [np.nan, 0.64815],
            [4,      0.814815],
            [5,      0.981482],
        ],
        columns=cols,
        index=idx,
    )

    assert dfs_close(got.df, expected_df)
    assert got.transitions is None
