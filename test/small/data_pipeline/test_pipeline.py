import numpy as np
import pandas as pd
import datetime

from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import CallerCode
from corerl.config import MainConfig
from corerl.data_pipeline.transition_creators.dummy import DummyTransitionCreatorConfig


def test_construct_pipeline():
    cfg = MainConfig(
        tags=[
            TagConfig(name='sensor_x'),
            TagConfig(name='sensor_y'),
        ],
        agent_transition_creator=DummyTransitionCreatorConfig(),
    )
    _ = Pipeline(cfg)


def test_passing_data_to_pipeline():
    cfg = MainConfig(
        tags=[
            TagConfig(name='sensor_x'),
            TagConfig(name='sensor_y'),
        ],
        agent_transition_creator=DummyTransitionCreatorConfig(),
    )
    pipeline = Pipeline(cfg)

    cols = {"sensor_x": [np.nan, 1.0], "sensor_y": [2.0, np.nan]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, 1 ),
        datetime.datetime(2024, 1, 1, 1, 2)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)

    # test that we can run the pf through the pipeline
    _ = pipeline(df, caller_code=CallerCode.OFFLINE)
