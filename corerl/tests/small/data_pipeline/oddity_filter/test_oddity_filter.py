import numpy as np
import pandas as pd
from test.infrastructure.utils.pandas import dfs_close

from corerl.configs.data_pipeline.oddity_filters.config import GlobalOddityFilterConfig
from corerl.configs.data_pipeline.oddity_filters.ema_filter import EMAFilterConfig
from corerl.configs.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.configs.data_pipeline.oddity_filters.stuck_detector import StuckDetectorConfig
from corerl.configs.tags.tag_config import BasicTagConfig
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.oddity_filters.oddity_filter import OddityFilterConstructor
from corerl.state import AppState
from tests.sdk.factories import PipelineFrameFactory


def test_oddity0(dummy_app_state: AppState):
    tags = [BasicTagConfig(name="obs_1"), BasicTagConfig(name="obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[IdentityFilterConfig()])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    pf = PipelineFrameFactory.build(
        data={
            "obs_1": [np.nan, 1, 1, 1, np.nan, 5, 1],
            "obs_2": [1, 1, 1, np.nan, 1, -5, np.nan],
        },
        data_mode=DataMode.REFRESH,
    )

    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 1, 1, np.nan, 5, 1],
            "obs_2": [1, 1, 1, np.nan, 1, -5, np.nan],
        },
    )

    assert dfs_close(pf.data, expected_data)

def test_oddity1(dummy_app_state: AppState):
    tags = [BasicTagConfig(name="obs_1"), BasicTagConfig(name="obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[EMAFilterConfig(warmup=3)])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    pf = PipelineFrameFactory.build(
        data={
            "obs_1": [np.nan, 1, 1, 1, np.nan, 5, 1],
            "obs_2": [1, 1, 1, np.nan, 1, -5, np.nan],
        },
        data_mode=DataMode.REFRESH,
    )

    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0, 1.0, 1.0, np.nan, np.nan, 1.0],
            "obs_2": [1.0, 1.0, 1.0, np.nan, 1.0, np.nan, np.nan],
        },
    )

    assert dfs_close(pf.data, expected_data)

def test_oddity2(dummy_app_state: AppState):
    tags = [BasicTagConfig(name="obs_1"), BasicTagConfig(name="obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[EMAFilterConfig(warmup=3)])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    pf = PipelineFrameFactory.build(
        data={
            "obs_1": [np.nan, 1, 1, 1, np.nan],
            "obs_2": [1, 1, 1, np.nan, -5],
        },
        data_mode=DataMode.REFRESH,
    )

    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0, 1.0, 1.0, np.nan],
            "obs_2": [1.0, 1.0, 1.0, np.nan, np.nan],
        },
    )
    assert dfs_close(pf.data, expected_data)

    pf = PipelineFrameFactory.build(
        data={
            "obs_1": [5., 1.],
            "obs_2": [1, np.nan],
        },
        data_mode=DataMode.REFRESH,
        temporal_state=pf.temporal_state,
    )
    pf = oddity_filter(pf)
    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0],
            "obs_2": [1.0, np.nan],
        },
    )

    assert dfs_close(pf.data, expected_data)


def test_oddity3(dummy_app_state: AppState):
    """
    Same as oddity filter 2, with identity filter added to the chain of oddity detectors
    """

    tags = [BasicTagConfig(name="obs_1"), BasicTagConfig(name="obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[EMAFilterConfig(warmup=3), IdentityFilterConfig()])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    pf = PipelineFrameFactory.build(
        data={
            "obs_1": [np.nan, 1, 1, 1, np.nan],
            "obs_2": [1, 1, 1, np.nan, -5],
        },
        data_mode=DataMode.REFRESH,
    )

    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0, 1.0, 1.0, np.nan],
            "obs_2": [1.0, 1.0, 1.0, np.nan, np.nan],
        },
    )
    assert dfs_close(pf.data, expected_data)

    pf = PipelineFrameFactory.build(
        data={
            "obs_1": [5., 1.],
            "obs_2": [1, np.nan],
        },
        data_mode=DataMode.REFRESH,
        temporal_state=pf.temporal_state,
    )
    pf = oddity_filter(pf)
    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0],
            "obs_2": [1.0, np.nan],
        },
    )

    assert dfs_close(pf.data, expected_data)


def test_stuck1(dummy_app_state: AppState):
    """
    Same as oddity filter 2, with identity filter added to the chain of oddity detectors
    """

    tags = [BasicTagConfig(name="obs_1"), BasicTagConfig(name="obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[StuckDetectorConfig(step_tol=2)])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    pf = PipelineFrameFactory.build(
        data={
            "obs_1": [np.nan, 1, 1, 1, np.nan],
            "obs_2": [1, 1, 2, np.nan, 1],
        },
        data_mode=DataMode.REFRESH,
    )

    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0, 1.0, np.nan, np.nan],
            "obs_2": [1.0, 1.0, 2.0, np.nan, 1.0],
        },
    )

    assert dfs_close(pf.data, expected_data)


def test_stuck_and_ema(dummy_app_state: AppState):
    """
    Same as oddity filter 2, with identity filter added to the chain of oddity detectors
    """

    tags = [BasicTagConfig(name="obs_1"), BasicTagConfig(name="obs_2")]
    oddity_filter_cfg = GlobalOddityFilterConfig(defaults=[StuckDetectorConfig(step_tol=2), EMAFilterConfig(warmup=3)])
    oddity_filter = OddityFilterConstructor(tags, dummy_app_state, oddity_filter_cfg)

    pf = PipelineFrameFactory.build(
        data={
            "obs_1": [np.nan, 1, 2, 1.5, 1, 1, 1], # last 1 is stuck
            "obs_2": [1, 1, 1, np.nan, 2, 1.5, -5],
        },
        data_mode=DataMode.REFRESH,
    )

    pf = oddity_filter(pf)

    expected_data = pd.DataFrame(
        {
            "obs_1": [np.nan, 1.0, 2.0, 1.5, 1., 1., np.nan],
            "obs_2": [1.0, 1.0, np.nan, np.nan, 2.0, 1.5, np.nan],
        },
    )

    assert dfs_close(pf.data, expected_data)
