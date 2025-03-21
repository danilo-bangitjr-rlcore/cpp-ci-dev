import datetime as dt

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms.delta import DeltaConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig


def test_only_normalize_preprocess():
    tag_cfg1 = TagConfig(
        name="tag_1",
        operating_range=[0, 10],
        preprocess=[
            NormalizerConfig(min=0, max=10)
        ]
    )
    preprocessor = Preprocessor([tag_cfg1])

    df = pd.DataFrame({"tag_1": [1, 5]})
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = preprocessor(pf)
    out = pf.data["tag_1"].to_numpy()
    expected = np.array([0.1, 0.5])

    assert np.allclose(out, expected)

def test_only_delta_preprocess():
    tag_cfg1 = TagConfig(
        name="tag_1",
        preprocess=[
            DeltaConfig(time_thresh=dt.timedelta(hours=2))
        ]
    )
    preprocessor = Preprocessor([tag_cfg1])

    df = pd.DataFrame({"tag_1": [1, 2, np.nan, 3, np.nan, np.nan, np.nan, 4]})
    start_time = dt.datetime(2023, 7, 13, 5)
    increment = dt.timedelta(hours=1)
    num_inds = len(df)
    timestamps = []
    for i in range(num_inds):
        timestamps.append(start_time + i * increment)
    df.index = pd.DatetimeIndex(timestamps)

    pf = PipelineFrame(df, DataMode.ONLINE)
    pf = preprocessor(pf)
    out = pf.data["tag_1"].to_numpy()
    expected = np.array([np.nan, 1, np.nan, 1, np.nan, np.nan, np.nan, np.nan])

    assert np.allclose(out, expected, equal_nan=True)


def test_delta_norm_preprocess():
    tag_cfg1 = TagConfig(
        name="tag_1",
        preprocess=[
            DeltaConfig(time_thresh=dt.timedelta(hours=2)),
            NormalizerConfig(min=0, max=10)
        ]
    )
    preprocessor = Preprocessor([tag_cfg1])

    df = pd.DataFrame({"tag_1": [1, 2, np.nan, 3, np.nan, np.nan, 4]})
    start_time = dt.datetime(2023, 7, 13, 5)
    increment = dt.timedelta(hours=1)
    num_inds = len(df)
    timestamps = []
    for i in range(num_inds):
        timestamps.append(start_time + i * increment)
    df.index = pd.DatetimeIndex(timestamps)

    pf = PipelineFrame(df, DataMode.ONLINE)
    pf = preprocessor(pf)
    out = pf.data["tag_1"].to_numpy()
    expected = np.array([np.nan, 0.1, np.nan, 0.1, np.nan, np.nan, np.nan])

    assert np.allclose(out, expected, equal_nan=True)
