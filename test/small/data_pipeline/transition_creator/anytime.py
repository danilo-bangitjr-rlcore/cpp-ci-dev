import numpy as np
import pandas as pd
import datetime

from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.tag_config import TagConfig
from corerl.config import MainConfig
from corerl.data_pipeline.transition_creators.dummy import DummyTransitionCreatorConfig
from corerl.data_pipeline.datatypes import PipelineFrame

from corerl.data_pipeline.transition_creators.anytime import (
    AnytimeTransitionCreator,
    AnytimeTransitionCreatorConfig)
import pytest


@pytest.fixture
def pf():
    cols = {"action": [0, 0, 1, 1], "state": [1, 2, 3, 4]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, 1),
        datetime.datetime(2024, 1, 1, 1, 2),
        datetime.datetime(2024, 1, 1, 1, 3),
        datetime.datetime(2024, 1, 1, 1, 4)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(df)
    pf.action_tags = ['action']
    return pf


def test_anytime_inner_call_1(pf):
    cfg = AnytimeTransitionCreatorConfig()
    tc = AnytimeTransitionCreator(cfg)
    tc(pf)

