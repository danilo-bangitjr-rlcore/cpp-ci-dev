import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms.delta import Delta, DeltaConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from test.infrastructure.utils.pandas import dfs_close


def test_delta_xform():
    cfg = DeltaConfig()
    xform = Delta(cfg)

    df = pd.DataFrame({
        'tag-1_a': [1, 2, 3],
        'tag-1_b': [np.nan, 4, 6],
    })

    carry = TransformCarry(df, df, 'tag-1')
    carry, _ = xform(carry, None)

    expected = pd.DataFrame({
        'tag-1_a_delta': [np.nan, 1, 1],
        'tag-1_b_delta': [np.nan, np.nan, 2],
    })
    assert dfs_close(carry.transform_data, expected)

def test_delta_xform_ts():
    cfg = DeltaConfig()
    xform = Delta(cfg)

    df1 = pd.DataFrame({
        'tag-1_a': [1, 2, 3],
        'tag-1_b': [np.nan, 4, 6],
    })

    carry = TransformCarry(df1, df1, 'tag-1')
    carry, ts = xform(carry, None)

    df2 = pd.DataFrame({
        'tag-1_a': [5, 8, 12],
        'tag-1_b': [8, 10, 10],
    })
    carry = TransformCarry(df2, df2, 'tag-1')
    carry, _ = xform(carry, ts)

    expected = pd.DataFrame({
        # df1 -> df2
        # [1, 2, 3] -> [5, 8, 12]
        'tag-1_a_delta': [2, 3, 4],
        'tag-1_b_delta': [2, 2, 0],
    })

    assert dfs_close(carry.transform_data, expected)
