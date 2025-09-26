import datetime

import numpy as np
import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.state import AppState


def make_time_index(start: datetime.datetime, steps: int, delta: datetime.timedelta):
    return pd.DatetimeIndex([start + i * delta for i in range(steps)])


@pytest.fixture
def dummy_pipeline(dummy_app_state: AppState, basic_config: MainConfig) -> Pipeline:
    # Use basic_config.pipeline for pipeline config
    pipeline_cfg = basic_config.pipeline
    return Pipeline(dummy_app_state, pipeline_cfg)


@pytest.fixture
def fake_pipeline_data():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)
    idx = make_time_index(start, 7, Δ)
    cols = ['tag-1', 'tag-2', 'reward', 'action-1']
    return pd.DataFrame(
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

@pytest.fixture
def fake_clean_pipeline_data():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)
    idx = make_time_index(start, 7, Δ)
    cols = ['tag-1', 'tag-2', 'reward', 'action-1']
    return pd.DataFrame(
        data=[
            [0,      0,        0,    0],
            [0,      2,        3,    1],
            [1,      4,        0,    0],
            [2,      6,        0,    1],
            [3,      8,        0,    0],
            [4,      10,       1,    1],
            [5,      12,       0,    0],
        ],
        columns=cols,
        index=idx,
    )
