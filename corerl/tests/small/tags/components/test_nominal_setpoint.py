import os

import jax.numpy as jnp
from lib_config.errors import ConfigValidationErrors

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.configs.tags.setpoint import SetpointTagConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState
from tests.infrastructure.config import create_config_with_overrides


def test_nominal_setpoint_norm():
    config_path = os.path.join(os.path.dirname(__file__), 'assets/nominal_setpoint.yaml')
    cfg = create_config_with_overrides(base_config_path=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    assert isinstance(cfg.pipeline.tags[0], SetpointTagConfig)
    assert cfg.pipeline.tags[0].nominal_setpoint == 0.4

    event_bus = DummyEventBus()
    app_state = AppState[DummyEventBus, MainConfig](
        cfg=cfg,
        metrics=create_metrics_writer(cfg.metrics),
        evals=create_evals_writer(cfg.evals),
        event_bus=event_bus,
    )
    pipeline = Pipeline(app_state, cfg.pipeline)
    column_desc = pipeline.column_descriptions
    agent = GreedyAC(
        cfg.agent,
        app_state,
        column_desc,
    )

    assert jnp.isclose(agent._nominal_setpoints, jnp.array([0.4]))
