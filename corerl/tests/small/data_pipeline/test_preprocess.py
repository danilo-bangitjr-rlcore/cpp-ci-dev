import numpy as np
import pandas as pd
from test.infrastructure.utils.pandas import dfs_close

from corerl.configs.tags.tag_config import BasicTagConfig
from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.transforms.norm import NormalizerConfig


def test_only_normalize_preprocess():
    tag_cfg1 = BasicTagConfig(
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

def test_tag_names_same_prefix_preprocess():
    tag_cfg1 = BasicTagConfig(
        name="tag_1",
        operating_range=(0, 10),
        preprocess=[
            NormalizerConfig(min=0, max=10),
        ],
    )
    tag_cfg2 = BasicTagConfig(
        name="tag_1_sp",
        operating_range=(0, 10),
        preprocess=[],
    )
    preprocessor = Preprocessor([tag_cfg1, tag_cfg2])

    df = pd.DataFrame({"tag_1": [1, 5], "tag_1_sp": [1, 5]})
    pf = PipelineFrame(df, DataMode.ONLINE)

    pf = preprocessor(pf)
    out = pf.data

    expected = pd.DataFrame({"tag_1": [0.1, 0.5], "tag_1_sp": [1, 5]})

    assert dfs_close(out, expected)
