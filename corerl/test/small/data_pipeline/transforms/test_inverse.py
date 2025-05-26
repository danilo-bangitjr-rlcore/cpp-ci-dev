import datetime

import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.inverse import Inverse, InverseConfig
from test.infrastructure.utils.pandas import dfs_close


def test_inverse():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [1e-4, 4],
            [1e-2, 6],
            [np.nan, np.nan],
            [4, 10],
            [5, 12],
        ],
        columns=cols,
        index=idx,
    )

    tf = Inverse(InverseConfig(tolerance=1e-3))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["1/tag-1"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [np.nan],
            [np.nan],
            [100.0],
            [np.nan],
            [0.25],
            [0.2],
        ],
        columns=expected_cols,
        index=idx,
    )
    print(tf_carry.transform_data, expected_df)

    assert dfs_close(tf_carry.transform_data, expected_df)

