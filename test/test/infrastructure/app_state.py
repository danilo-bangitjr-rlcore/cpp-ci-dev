import pytest
from corerl.config import MainConfig
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState
from lib_config.loader import direct_load_config


def make_dummy_app_state(cfg: MainConfig):
    cfg.metrics.enabled = False
    cfg.evals.enabled = False
    return AppState(
        cfg=cfg,
        metrics=create_metrics_writer(cfg.metrics),
        evals=create_evals_writer(cfg.evals),
        event_bus=DummyEventBus(),
    )

@pytest.fixture()
def dummy_app_state() -> AppState:
    cfg = direct_load_config(MainConfig, config_name='../config/dep_mountain_car_continuous.yaml')
    assert isinstance(cfg, MainConfig)

    return make_dummy_app_state(cfg)
