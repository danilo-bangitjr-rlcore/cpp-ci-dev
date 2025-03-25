from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from corerl.data_pipeline.bound_checker import bound_checker
from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.tag_config import FloatBounds, TagConfig


@dataclass
class Case:
    bounds: dict[str, FloatBounds]
    raw: dict[str, list[float]]
    expected: dict[str, list[bool]]


@pytest.mark.parametrize(
    "case",
    [
        Case(
            bounds={   'tag-1': (None, None),          'tag-2': (None, None) },
            raw={      'tag-1': [3.4, -0.2, 2.7],      'tag-2': [-0.4, 6.3, -3.8] },
            expected={ 'tag-1': [False, False, False], 'tag-2': [False, False, False] },
        ),
        Case(
            bounds={   'tag-1': (0, 10),               'tag-2': (-1, 10) },
            raw={      'tag-1': [3.4, -0.2, 2.7],      'tag-2': [-0.4, 6.3, -3.8] },
            expected={ 'tag-1': [False, True, False],  'tag-2': [False, False, True] },
        ),
        Case(
            bounds={   'tag-1': (0, 1),                'tag-2': (-1, 10) },
            raw={      'tag-1': [0.4, 1.3, 0.7],       'tag-2': [11.9, -0.5, 3.6] },
            expected={ 'tag-1': [False, True, False],  'tag-2': [True, False, False] },
        ),
    ]
)
def test_bounds(case: Case):
    tag_cfgs = [
        TagConfig(
            name=key,
            operating_range=bound,
            preprocess=[],
        )
        for key, bound in case.bounds.items()
    ]

    prep = Preprocessor(tag_cfgs)
    data = pd.DataFrame(case.raw)
    pf = PipelineFrame(data, DataMode.ONLINE)

    for cfg in tag_cfgs:
        assert cfg.operating_range is not None
        pf = bound_checker(pf, cfg.name, cfg.operating_range, prep)

    for key, expect in case.expected.items():
        assert np.all(pf.data[key].isna() == expect)
