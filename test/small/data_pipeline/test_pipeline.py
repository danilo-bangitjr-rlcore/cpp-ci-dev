import numpy as np
import pandas as pd

from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.tag_config import TagConfig


def test_construct_pipeline():
    pl_cfg = PipelineConfig()
    _ = Pipeline(pl_cfg)


def test_passing_data_to_pipeline():
    pl_cfg = PipelineConfig()
    pipeline = Pipeline(pl_cfg)

    cols = {"sensor_x": [np.nan, 1.0], "sensor_y": [2.0, np.nan]}
    data = pd.DataFrame(cols)

    # test that we can run the pf through the pipeline
    tag_cfg = TagConfig()
    _ = pipeline(data, tag_cfg)
