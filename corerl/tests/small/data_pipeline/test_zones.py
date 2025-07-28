from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from lib_config.loader import direct_load_config

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.zones import ZoneDiscourager, ZoneViolationEvent
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.messages.events import RLEventType
from corerl.state import AppState


@pytest.fixture
def cfg():
    return direct_load_config(
        MainConfig,
        config_name='tests/small/data_pipeline/assets/zone.yaml',
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


@pytest.mark.parametrize('data_mode', [DataMode.OFFLINE, DataMode.ONLINE])
def test_zones1(
    cfg: MainConfig,
    app_state: AppState[DummyEventBus],
    pipeline: Pipeline,
    data_mode: DataMode,
):
    assert cfg.pipeline.reward is not None

    zone_discourager = ZoneDiscourager(app_state, cfg.pipeline.tags, pipeline.preprocessor)

    start = datetime.now(UTC)
    Δ = timedelta(minutes=5)

    dates = [start + i * Δ for i in range(3)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-0', 'tag-1', 'tag-2']
    df = pd.DataFrame(
        data=[
            # tag-0 red zone full violation (note no yellow zone specified)
            # tag-1 red zone full violation
            # tag-1 yellow zone full violation
            # tag-2 yellow zone full violation
            [10.,   0.,    0.],
            # tag-1 yellow zone full violation
            [4,     9,     7],
            # tag-2 yellow zone partial violation
            [4,     6,     0.25],
        ],
        columns=cols,
        index=idx,
    )

    pf = PipelineFrame(df, data_mode=data_mode)
    pf = pipeline.preprocessor(pf)
    pf.rewards = pd.DataFrame({ 'reward': [-1.] * 3 }, index=idx)

    pf = zone_discourager(pf)
    emitted_events = app_state.event_bus.get_last_events()

    if pf.data_mode == DataMode.OFFLINE:
        assert len(emitted_events) == 0, 'Expected no events to be emitted in offline mode'
    elif pf.data_mode == DataMode.ONLINE:
        assert len(emitted_events) == 6
        assert (
            isinstance(emitted_events[0], ZoneViolationEvent)
            and emitted_events[0].type == RLEventType.red_zone_violation
            and emitted_events[0].tag == 'tag-0'
        )
        assert (
            isinstance(emitted_events[1], ZoneViolationEvent)
            and emitted_events[1].type == RLEventType.red_zone_violation
            and emitted_events[1].tag == 'tag-1'
        )
        assert (
            isinstance(emitted_events[2], ZoneViolationEvent)
            and emitted_events[2].type == RLEventType.yellow_zone_violation
            and emitted_events[2].tag == 'tag-1'
        )
        assert (
            isinstance(emitted_events[3], ZoneViolationEvent)
            and emitted_events[3].type == RLEventType.yellow_zone_violation
            and emitted_events[3].tag == 'tag-2'
        )
        assert (
            isinstance(emitted_events[4], ZoneViolationEvent)
            and emitted_events[4].type == RLEventType.yellow_zone_violation
            and emitted_events[4].tag == 'tag-1'
        )
        assert (
            isinstance(emitted_events[5], ZoneViolationEvent)
            and emitted_events[5].type == RLEventType.yellow_zone_violation
            and emitted_events[5].tag == 'tag-2'
        )

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
