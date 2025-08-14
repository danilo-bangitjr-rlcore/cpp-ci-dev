import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.state import AppState


def make_time_index(start: datetime.datetime, steps: int, delta: datetime.timedelta):
    return pd.DatetimeIndex([start + i * delta for i in range(steps)])


@pytest.fixture
def fake_pipeline1_data():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)
    idx = make_time_index(start, 7, Δ)
    cols = ['tag-1', 'tag-2', 'reward', 'action-1']
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
    return df, idx


@pytest.fixture
def fake_pipeline1_single_row():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)
    idx = make_time_index(start, 1, Δ)
    cols = ['tag-1', 'tag-2', 'reward', 'action-1']
    df = pd.DataFrame(
        data=[
            [0, 1, 2, 3],
        ],
        columns=cols,
        index=idx,
    )
    return df, idx


def test_pipeline_benchmark(
    benchmark: Any,
    dummy_app_state: AppState,
    basic_config: MainConfig,
    fake_pipeline1_data: tuple[pd.DataFrame, pd.DatetimeIndex],
):
    pipeline = Pipeline(dummy_app_state, basic_config.pipeline)
    df, _ = fake_pipeline1_data

    def _inner(pipeline: Pipeline, df: pd.DataFrame):
        pipeline(df)

    benchmark(_inner, pipeline, df)


def test_pipeline_benchmark_single_row(
    benchmark: Any,
    dummy_app_state: AppState,
    basic_config: MainConfig,
    fake_pipeline1_single_row: tuple[pd.DataFrame, pd.DatetimeIndex],
):
    pipeline = Pipeline(dummy_app_state, basic_config.pipeline)
    df, _ = fake_pipeline1_single_row

    def _inner(pipeline: Pipeline, df: pd.DataFrame):
        pipeline(df)

    benchmark(_inner, pipeline, df)
