import datetime

import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.power import Power, PowerConfig
from test.infrastructure.utils.pandas import dfs_close


def test_power_xform():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [1, 4],
            [2, 6],
            [np.nan, np.nan],
            [4, 10],
            [5, 12],
        ],
        columns=cols,
        index=idx,
    )

    tf = Power(PowerConfig(pow=2))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-1^2.0"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [0],
            [1],
            [4],
            [np.nan],
            [16],
            [25],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)


