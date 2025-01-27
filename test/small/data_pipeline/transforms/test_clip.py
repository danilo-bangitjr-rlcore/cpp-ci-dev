import pandas as pd

from corerl.data_pipeline.transforms.clip import Clip, ClipConfig
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

    cfg = ClipConfig(bounds=(3, 5))
    xform = Clip(cfg)

    carry, _ = xform(carry, None)

    expected_df = pd.DataFrame({
        'tag-1_clip': [3., 3, 3, 4, 5, 5],
    })
    pd.testing.assert_frame_equal(carry.transform_data, expected_df)
