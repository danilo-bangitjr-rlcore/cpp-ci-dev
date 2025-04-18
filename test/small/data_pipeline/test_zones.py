from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.zones import ZoneDiscourager
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState


@pytest.fixture
def cfg():
    return direct_load_config(
        MainConfig,
        base='test/small/data_pipeline/assets',
        config_name='zone.yaml',
    )


@pytest.fixture
def app_state(cfg: MainConfig):
    cfg.metrics.enabled = False
    cfg.evals.enabled = False
    return AppState(
        cfg=cfg,
        evals=EvalsTable(cfg.evals),
        metrics=MetricsTable(cfg.metrics),
        event_bus=DummyEventBus(),
    )


@pytest.fixture
def pipeline(cfg: MainConfig, app_state: AppState):
    return Pipeline(app_state, cfg.pipeline)


def test_zones1(cfg: MainConfig, app_state: AppState, pipeline: Pipeline):
    assert cfg.pipeline.reward is not None

    zone_discourager = ZoneDiscourager(app_state, cfg.pipeline.tags, pipeline.preprocessor)

    start = datetime.now(UTC)
    Δ = timedelta(minutes=5)

    dates = [start + i * Δ for i in range(3)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-0', 'tag-1', 'tag-2']
    df = pd.DataFrame(
        data=[
            # tag-0 red zone full violation
            [10.,   0.,    0.],
            # tag-1 yellow zone full violation
            [4,     9,     7],
            # tag-2 yellow zone partial violation
            [4,     6,     0.25],
        ],
        columns=cols,
        index=idx,
    )

    pf = PipelineFrame(df, data_mode=DataMode.OFFLINE)
    pf = pipeline.preprocessor(pf)
    pf.rewards = pd.DataFrame({ 'reward': [-1.] * 3 }, index=idx)

    pf = zone_discourager(pf)

    expected_rewards = pd.DataFrame(
        index=idx,
        columns=['reward'],
        data=[
            [-8],
            [-3],
            [-2.125],
        ],
    )

    pd.testing.assert_frame_equal(pf.rewards, expected_rewards)
