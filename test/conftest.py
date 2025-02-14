import pytest

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.eval.evals import evals_group
from corerl.eval.metrics import metrics_group
from corerl.messages.event_bus import EventBus
from corerl.state import AppState


@pytest.fixture()
def dummy_app_state() -> AppState:
    cfg = direct_load_config(MainConfig, config_name='config/dep_mountain_car_continuous.yaml', base='.')
    cfg.metrics.enabled = False
    cfg.evals.enabled = False
    cfg.event_bus.enabled = False
    event_bus = EventBus(cfg.event_bus, cfg.env)
    app_state = AppState(
        cfg=cfg,
        metrics=metrics_group.dispatch(cfg.metrics),
        evals=evals_group.dispatch(cfg.evals),
        event_bus=event_bus,
    )
    return app_state


pytest_plugins = (
    "test.infrastructure.networking",
    "test.infrastructure.utils.docker",
    "test.infrastructure.utils.pandas",
    "test.infrastructure.utils.tsdb",
)
