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
        operating_range=(0, 10),
        preprocess=[
            NormalizerConfig(min=0, max=10),
        ],
    )
    preprocessor = Preprocessor([tag_cfg1])

    df = pd.DataFrame({"tag_1": [1, 5]})
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = preprocessor(pf)
    out = pf.data["tag_1"].to_numpy()
    expected = np.array([0.1, 0.5])

    assert np.allclose(out, expected)
