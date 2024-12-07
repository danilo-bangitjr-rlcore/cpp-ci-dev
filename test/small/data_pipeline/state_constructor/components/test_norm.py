import numpy as np
import pandas as pd
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.state_constructors.sc import transform_group

from test.infrastructure.utils.pandas import dfs_close

def test_norm_part():
    raw_obs = pd.DataFrame({
        'obs-1': [np.nan, 0, 2, 4, 6, 8, 10, np.nan],
        'obs-2': [0, 1, 2, np.nan, 3, 4, np.nan, 5],
    })

    carry = TransformCarry(
        obs=raw_obs,
        transform_data=raw_obs,
        tag='obs',
    )

    norm_cfg = NormalizerConfig()
    normalizer = transform_group.dispatch(norm_cfg)

    new_carry, _ = normalizer(carry, None)

    expected = pd.DataFrame({
        'obs-1_norm': [np.nan, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan],
        'obs-2_norm': [0, 0.2, 0.4, np.nan, 0.6, 0.8, np.nan, 1.0],
    })

    assert dfs_close(new_carry.transform_data, expected)
