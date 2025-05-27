import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms.bounds import BoundsConfig, BoundsXform
from corerl.data_pipeline.transforms.interface import TransformCarry


def test_clip():
    df = pd.DataFrame({
        'tag-1': [1, 2, 3, 4, 5, 6],
    })

    carry = TransformCarry(
        obs=df,
        transform_data=df,
        tag='tag-1',
    )

    cfg = BoundsConfig(bounds=(3, 5))
    xform = BoundsXform(cfg)

    carry, _ = xform(carry, None)

    expected_df = pd.DataFrame({
        'tag-1_bounds': [3., 3, 3, 4, 5, 5],
    })
    pd.testing.assert_frame_equal(carry.transform_data, expected_df)


def test_nan():
    df = pd.DataFrame({
        'tag-1': [1, 2, 3, 4, 5, 6],
    })

    carry = TransformCarry(
        obs=df,
        transform_data=df,
        tag='tag-1',
    )

    cfg = BoundsConfig(bounds=(3, 5), mode='nan')
    xform = BoundsXform(cfg)

    carry, _ = xform(carry, None)

    expected_df = pd.DataFrame({
        'tag-1_bounds': [np.nan, np.nan, 3, 4, 5, np.nan],
    })
    pd.testing.assert_frame_equal(carry.transform_data, expected_df)
