import datetime

import pandas as pd
import pytest
from test.infrastructure.networking import get_free_port

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.messages.event_bus import EventBus
from corerl.messages.events import RLEventType
from corerl.state import AppState
from tests.infrastructure.config import ConfigBuilder, create_config_with_overrides


@pytest.fixture(scope="function")
def main_cfg():
    return create_config_with_overrides(
        base_config_path='tests/large/events/assets/trigger.yaml',
        overrides={'event_bus.enabled': True},
    )


def test_no_trigger(main_cfg: MainConfig):
    port = get_free_port('localhost')
    cfg = ConfigBuilder(main_cfg).with_override('event_bus.cli_connection', f'tcp://localhost:{port}').build()

    event_bus = EventBus(cfg.event_bus)
    event_bus.start()

    app_state = AppState(
        cfg=cfg,
        metrics=create_metrics_writer(main_cfg.metrics),
        evals=create_evals_writer(main_cfg.evals),
        event_bus=event_bus,
    )

    pipeline = Pipeline(app_state, main_cfg.pipeline)

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(4)]
    idx = pd.DatetimeIndex(dates)

    df = pd.DataFrame({
        'action-0': [1, 2, 3, 4],
        'tag-0': [1, 1, 1, 1],
        'reward': [0, 1, 2, 3],
        'terminated': [False, False, False, False],
        'truncated': [False, False, False, False],
    }, index=idx)
    pipeline(df, data_mode=DataMode.ONLINE)

    event = event_bus.recv_event()
    assert event is None

    event_bus.cleanup()

def test_trigger(main_cfg: MainConfig):
    port = get_free_port('localhost')
    cfg = ConfigBuilder(main_cfg).with_override('event_bus.cli_connection', f'tcp://localhost:{port}').build()

    event_bus = EventBus(cfg.event_bus)
    event_bus.start()

    app_state = AppState(
        cfg=main_cfg,
        metrics=create_metrics_writer(main_cfg.metrics),
        evals=create_evals_writer(main_cfg.evals),
        event_bus=event_bus,
    )

    pipeline = Pipeline(app_state, main_cfg.pipeline)

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(4)]
    idx = pd.DatetimeIndex(dates)

    df = pd.DataFrame({
        'action-0': [1, 2, 3, 4],
        'tag-0': [1, 1, 1, 2],
        'reward': [0, 1, 2, 3],
        'terminated': [False, False, False, False],
        'truncated': [False, False, False, False],
    }, index=idx)
    pipeline(df, data_mode=DataMode.ONLINE)

    event = event_bus.recv_event()
    assert event is not None
    assert event.type == RLEventType.action_period_reset

    event_bus.cleanup()
