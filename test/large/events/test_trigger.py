import datetime

import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import EventBus
from corerl.messages.events import EventType
from corerl.state import AppState
from test.infrastructure.networking import get_free_port


@pytest.fixture(scope="function")
def main_cfg():
    cfg = direct_load_config(
        MainConfig,
        base='test/large/events/assets',
        config_name='trigger.yaml',
    )
    assert isinstance(cfg, MainConfig)

    cfg.event_bus.enabled = True
    return cfg


def test_no_trigger(
    main_cfg: MainConfig,
):
    port = get_free_port('localhost')
    main_cfg.event_bus.cli_connection = f'tcp://localhost:{port}'
    event_bus = EventBus(
        main_cfg.event_bus,
        main_cfg.env,
    )

    app_state = AppState(
        cfg=main_cfg,
        metrics=MetricsTable(main_cfg.metrics),
        evals=EvalsTable(main_cfg.evals),
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


def test_trigger(
    main_cfg: MainConfig,
):
    port = get_free_port('localhost')
    main_cfg.event_bus.cli_connection = f'tcp://localhost:{port}'
    event_bus = EventBus(
        main_cfg.event_bus,
        main_cfg.env,
    )
    event_bus.start()
    print("HELLO")
    print("HELLO")
    print("HELLO")
    print("HELLO")
    print("HELLO")
    print("HELLO")

    app_state = AppState(
        cfg=main_cfg,
        metrics=MetricsTable(main_cfg.metrics),
        evals=EvalsTable(main_cfg.evals),
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
    assert event.type == EventType.action_period_reset

    event_bus.cleanup()
