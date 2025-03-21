import datetime as dt

import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms.delta import Delta, DeltaConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from test.infrastructure.utils.pandas import dfs_close


def test_delta_xform():
    cfg = DeltaConfig(time_thresh=dt.timedelta(hours=2))
    xform = Delta(cfg)

    df = pd.DataFrame({
        'tag-1_a': [1, 2, np.nan, np.nan, 3],
        'tag-1_b': [np.nan, 4, np.nan, 6, np.nan],
    })
    start_time = dt.datetime(2023, 4, 11, 2)
    increment = dt.timedelta(hours=1)
    timestamps = []
    for i in range(len(df)):
        timestamps.append(start_time + increment * i)
    df.index = pd.DatetimeIndex(timestamps)

    carry = TransformCarry(df, df, 'tag-1')
    carry, _ = xform(carry, None)

    expected = pd.DataFrame({
        'tag-1_a_Δ': [np.nan, 1, np.nan, np.nan, np.nan],
        'tag-1_b_Δ': [np.nan, np.nan, np.nan, 2, np.nan],
    })
    assert dfs_close(carry.transform_data, expected)

def test_delta_xform_ts():
    cfg = DeltaConfig(time_thresh=dt.timedelta(hours=2))
    xform = Delta(cfg)

    df1 = pd.DataFrame({
        'tag-1_a': [1, np.nan, np.nan],
        'tag-1_b': [np.nan, 4, np.nan],
    })
    df1.index = pd.DatetimeIndex(["7/13/2023 4:00", "7/13/2023 5:00", "7/13/2023 6:00"])

    carry = TransformCarry(df1, df1, 'tag-1')
    carry, ts = xform(carry, None)

    df2 = pd.DataFrame({
        'tag-1_a': [5, 8, 12],
        'tag-1_b': [8, 10, 10],
    })
    df2.index = pd.DatetimeIndex(["7/13/2023 7:00", "7/13/2023 8:00", "7/13/2023 9:00"])
    carry = TransformCarry(df2, df2, 'tag-1')
    carry, _ = xform(carry, ts)

    expected = pd.DataFrame({
        # df1 -> df2
        # [1, 2, 3] -> [5, 8, 12]
        'tag-1_a_Δ': [np.nan, 3, 4],
        'tag-1_b_Δ': [4, 2, 0],
    })

    assert dfs_close(carry.transform_data, expected)
