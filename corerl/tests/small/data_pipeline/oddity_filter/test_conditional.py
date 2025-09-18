import os

import numpy as np
import pandas as pd
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config
from test.infrastructure.utils.pandas import dfs_close

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.oddity_filters.oddity_filter import OddityFilterConstructor
from corerl.state import AppState


def test_global_sympy_condition_filter(dummy_app_state: AppState):
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/sympy_condition_filter.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)

    oddity_filter = OddityFilterConstructor(cfg.pipeline.tags, dummy_app_state, cfg.pipeline.oddity_filter)

    raw_obs = pd.DataFrame(
        {
            "action-1": [1, 2, 3, 4, np.nan, 6, 7],
            "tag-1": [0, 1, 0, np.nan, 1, 1, 0],
            "tag-2": [1, 1, 0, np.nan, 0, 1, 0],
            "tag-3": [8, np.nan, 6, np.nan, 4, 3, 2],
        },
    )

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.ONLINE,
    )

    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "action-1": [1, 2, np.nan, 4, np.nan, 6, np.nan],
            "tag-1": [0, 1, 0, np.nan, 1, 1, 0],
            "tag-2": [1, 1, 0, np.nan, 0, 1, 0],
            "tag-3": [8, np.nan, np.nan, np.nan, 4, 3, np.nan],
        },
    )

    assert dfs_close(pf.data, expected_data)
