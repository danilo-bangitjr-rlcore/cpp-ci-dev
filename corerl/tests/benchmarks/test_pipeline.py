import datetime
from typing import Any

import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.state import AppState
from tests.infrastructure.pipeline import make_time_index


@pytest.fixture
def fake_pipeline_single_row():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)
    idx = make_time_index(start, 1, Δ)
    cols = ['tag-1', 'tag-2', 'reward', 'action-1']
    return pd.DataFrame(
        data=[
            [0, 1, 2, 3],
        ],
        columns=cols,
        index=idx,
    )


def test_pipeline_benchmark(
    benchmark: Any,
    dummy_app_state: AppState,
    basic_config: MainConfig,
    fake_pipeline_data: pd.DataFrame,
):
    pipeline = Pipeline(dummy_app_state, basic_config.pipeline)

    def _inner(pipeline: Pipeline, df: pd.DataFrame):
        pipeline(df)

    benchmark(_inner, pipeline, fake_pipeline_data)


def test_pipeline_benchmark_single_row(
    benchmark: Any,
    dummy_app_state: AppState,
    basic_config: MainConfig,
    fake_pipeline_single_row: pd.DataFrame,
):
    pipeline = Pipeline(dummy_app_state, basic_config.pipeline)

    def _inner(pipeline: Pipeline, df: pd.DataFrame):
        pipeline(df)

    benchmark(_inner, pipeline, fake_pipeline_single_row)
