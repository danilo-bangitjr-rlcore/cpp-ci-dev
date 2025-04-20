import pytest

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState


@pytest.fixture()
def dummy_app_state() -> AppState:
    cfg = direct_load_config(MainConfig, config_name='config/dep_mountain_car_continuous.yaml', base='.')
    assert isinstance(cfg, MainConfig)

    cfg.metrics.enabled = False
    cfg.evals.enabled = False
    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=DummyEventBus(),
    )
    return app_state
