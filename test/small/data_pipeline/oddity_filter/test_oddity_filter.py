import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.oddity_filters.config import GlobalOddityFilterConfig
from corerl.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.data_pipeline.oddity_filters.oddity_filter import OddityFilterConstructor
from corerl.data_pipeline.tag_config import TagConfig
from corerl.state import AppState
from test.infrastructure.utils.pandas import dfs_close


def test_oddity0(dummy_app_state: AppState):

    tags = [TagConfig("obs_1"), TagConfig("obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[IdentityFilterConfig()])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 1, 1, np.nan, 5, 1],
            "obs_2": [1, 1, 1, np.nan, 1, -5, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.REFRESH,
    )


    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 1, 1, np.nan, 5, 1],
            "obs_2": [1, 1, 1, np.nan, 1, -5, np.nan],
        }
    )

    assert dfs_close(pf.data, expected_data)

def test_oddity1(dummy_app_state: AppState):

    tags = [TagConfig("obs_1"), TagConfig("obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[EMAFilterConfig(warmup=3)])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 1, 1, np.nan, 5, 1],
            "obs_2": [1, 1, 1, np.nan, 1, -5, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.REFRESH,
    )


    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0, 1.0, 1.0, np.nan, np.nan, 1.0],
            "obs_2": [1.0, 1.0, 1.0, np.nan, 1.0, np.nan, np.nan],
        }
    )

    assert dfs_close(pf.data, expected_data)

def test_oddity2(dummy_app_state: AppState):

    tags = [TagConfig("obs_1"), TagConfig("obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[EMAFilterConfig(warmup=3)])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 1, 1, np.nan],
            "obs_2": [1, 1, 1, np.nan, -5],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.REFRESH,
    )


    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0, 1.0, 1.0, np.nan],
            "obs_2": [1.0, 1.0, 1.0, np.nan, np.nan],
        }
    )
    assert dfs_close(pf.data, expected_data)

    raw_obs = pd.DataFrame(
        {
            "obs_1": [5., 1.],
            "obs_2": [1, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.REFRESH,
        temporal_state=pf.temporal_state
    )
    pf = oddity_filter(pf)
    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0],
            "obs_2": [1.0, np.nan],
        }
    )

    assert dfs_close(pf.data, expected_data)


def test_oddity3(dummy_app_state: AppState):
    """
    Same as oddity filter 2, with identity filter added to the chain of oddity detectors
    """

    tags = [TagConfig("obs_1"), TagConfig("obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[EMAFilterConfig(warmup=3), IdentityFilterConfig()])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 1, 1, np.nan],
            "obs_2": [1, 1, 1, np.nan, -5],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.REFRESH,
    )


    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0, 1.0, 1.0, np.nan],
            "obs_2": [1.0, 1.0, 1.0, np.nan, np.nan],
        }
    )
    assert dfs_close(pf.data, expected_data)

    raw_obs = pd.DataFrame(
        {
            "obs_1": [5., 1.],
            "obs_2": [1, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        data_mode=DataMode.REFRESH,
        temporal_state=pf.temporal_state
    )
    pf = oddity_filter(pf)
    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0],
            "obs_2": [1.0, np.nan],
        }
    )

    assert dfs_close(pf.data, expected_data)
