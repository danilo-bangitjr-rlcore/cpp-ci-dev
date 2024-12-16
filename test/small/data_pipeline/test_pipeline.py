from typing import Any
import numpy as np
import pandas as pd
import datetime

from corerl.data_pipeline.imputers.linear import LinearImputerConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.state_constructors.sc import SCConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.data_pipeline.transforms.affine import AffineConfig
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
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
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
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
    )
    pipeline = Pipeline(cfg)

    cols = {"sensor_x": [np.nan, 1.0, 2.0], "sensor_y": [2.0, np.nan, 3.0]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, 1),
        datetime.datetime(2024, 1, 1, 1, 2),
        datetime.datetime(2024, 1, 1, 1, 3),
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)

    # test that we can run the pf through the pipeline
    _ = pipeline(df, caller_code=CallerCode.OFFLINE)


def test_state_action_dim():
    cfg = PipelineConfig(
        tags=[
            TagConfig(name='tag-1'),
            TagConfig(
                name='tag-2',
                bounds=(None, 10),
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    NormalizerConfig(),
                    TraceConfig(trace_values=[0.1, 0.9]),
                ],
            ),
            TagConfig(name='tag-3', is_action=True),
            TagConfig(name='tag-4', is_action=True),
        ],
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
        obs_interval_minutes=5,
    )

    pipeline = Pipeline(cfg)

    state_dim, action_dim = pipeline.get_state_action_dims()
    assert state_dim == 3
    assert action_dim == 2


def test_sub_pipeline1():
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
        ],
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
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

    cols = ['tag-1', 'tag-2_norm_trace-0.1']
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
            TagConfig(
                name='tag-1',
                state_constructor=[],
            ),
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
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
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

    cols = ['tag-1', 'tag-2_norm_trace-0.1']
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

def test_sub_pipeline3():
    """
    Same as test_sub_pipeline1, but adds reward constructor.

    Since the reward constructor isn't explicitly added to any of
    the tag configs, this should have the same result as test_sub_pipeline1.
    """
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
        ],
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
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
        stages=(StageCode.BOUNDS, StageCode.ODDITY, StageCode.IMPUTER, StageCode.RC, StageCode.SC),
    )

    cols = ['tag-1', 'tag-2_norm_trace-0.1']
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

    print(got.df)
    assert dfs_close(got.df, expected_df)
    assert got.transitions is None


def test_sub_pipeline4():
    """
    Same as test_sub_pipeline1, but adds reward constructor.
    """
    cfg = PipelineConfig(
        tags=[
            TagConfig(
                name='tag-1',
                state_constructor=[],
                reward_constructor=[
                    AffineConfig(
                        scale=-1,
                        bias=5
                    ),
                ]
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
        ],
        state_constructor=SCConfig(
            countdown=CountdownConfig(action_period=1),
        ),
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
        stages=(StageCode.BOUNDS, StageCode.ODDITY, StageCode.IMPUTER, StageCode.RC, StageCode.SC),
    )

    cols = ['reward', 'tag-1', 'tag-2_norm_trace-0.1']
    expected_df = pd.DataFrame(
        data=[
            [np.nan, np.nan, 0],
            [5,      0,      0.18],
            [4,      1,      0.378],
            [3,      2,      0.5778],
            [np.nan, np.nan, 0.77778],
            [1,      4,      0.977778],
            [0,      5,      np.nan],
        ],
        columns=cols,
        index=idx,
    )

    print(got.df)
    assert dfs_close(got.df, expected_df)
    assert got.transitions is None
