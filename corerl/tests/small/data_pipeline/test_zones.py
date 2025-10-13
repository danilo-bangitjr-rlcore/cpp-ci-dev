import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.zones import ZoneDiscourager, ZoneViolationEvent
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.messages.event_bus import DummyEventBus
from corerl.messages.events import RLEventType
from corerl.state import AppState
from tests.infrastructure.config import create_config_with_overrides
from tests.sdk.factories import PipelineFrameFactory


@pytest.fixture
def cfg():
    return create_config_with_overrides(
        base_config_path='tests/small/data_pipeline/assets/zone.yaml',
        overrides={
            'metrics.enabled': False,
            'evals.enabled': False,
        },
    )

@pytest.fixture
def reflex_cfg():
    return create_config_with_overrides(
        base_config_path='tests/small/data_pipeline/assets/red_reflex.yaml',
        overrides={
            'metrics.enabled': False,
            'evals.enabled': False,
        },
    )

@pytest.fixture
def app_state(cfg: MainConfig):
    return AppState(
        cfg=cfg,
        evals=create_evals_writer(cfg.evals),
        metrics=create_metrics_writer(cfg.metrics),
        event_bus=DummyEventBus(),
    )


@pytest.fixture
def pipeline(cfg: MainConfig, app_state: AppState):
    return Pipeline(app_state, cfg.pipeline)

@pytest.fixture
def reflex_pipeline(reflex_cfg: MainConfig, app_state: AppState):
    return Pipeline(app_state, reflex_cfg.pipeline)


@pytest.mark.parametrize('data_mode', [DataMode.OFFLINE, DataMode.ONLINE])
def test_zones1(
    cfg: MainConfig,
    app_state: AppState[DummyEventBus, MainConfig],
    pipeline: Pipeline,
    data_mode: DataMode,
):
    assert cfg.pipeline.reward is not None

    zone_discourager = ZoneDiscourager(app_state, cfg.pipeline.tags, pipeline.preprocessor)

    pf = PipelineFrameFactory.build(
        data={
            'tag-0': [10., 4, 4],
            'tag-1': [0., 9, 6],
            'tag-2': [0., 7, 0.25],
        },
        data_mode=data_mode,
    )

    pf = pipeline.preprocessor(pf)
    pf.rewards = pd.DataFrame({ 'reward': [-1.] * 3 }, index=pf.data.index)

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
        index=pf.data.index,
        columns=['reward'],
        data=[
            [-8],
            [-3],
            [-2.125],
        ],
    )

    pd.testing.assert_frame_equal(pf.rewards, expected_rewards)


@pytest.mark.parametrize('data_mode', [DataMode.OFFLINE, DataMode.ONLINE])
def test_red_zone_reflex(
    reflex_cfg: MainConfig,
    app_state: AppState[DummyEventBus, MainConfig],
    reflex_pipeline: Pipeline,
    data_mode: DataMode,
):
    assert reflex_cfg.pipeline.reward is not None

    zone_discourager = ZoneDiscourager(app_state, reflex_cfg.pipeline.tags, reflex_pipeline.preprocessor)

    pf = PipelineFrameFactory.build(
        data={
            'action-1': [5., 5, 5],
            'tag-0': [10., 4, 4],
            'tag-1': [9., 0., 6],
            'tag-2': [0., 7, 0.25],
        },
        data_mode=data_mode,
    )

    pf.rewards = pd.DataFrame({ 'reward': [-1.] * 3 }, index=pf.data.index)
    pf = reflex_pipeline.preprocessor(pf)
    pf = reflex_pipeline.action_constructor(pf)
    pf = zone_discourager(pf)

    expected_a_lo = pd.DataFrame(
        index=pf.data.index,
        columns=['action-1-lo'],
        data=[
            [0.],
            [0.4],
            [0.],
        ],
    )

    # action bounds should be normalized
    expected_a_hi = pd.DataFrame(
        index=pf.data.index,
        columns=['action-1-hi'],
        data=[
            [0.1],
            [0.6],
            [1.],
        ],
    )

    pd.testing.assert_frame_equal(pf.action_lo, expected_a_lo)
    pd.testing.assert_frame_equal(pf.action_hi, expected_a_hi)
