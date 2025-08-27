import os

import jax.numpy as jnp
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState
from corerl.tags.setpoint import SetpointTagConfig


def test_nominal_setpoint_norm():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/nominal_setpoint.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    assert isinstance(cfg.pipeline.tags[0], SetpointTagConfig)
    assert cfg.pipeline.tags[0].nominal_setpoint == 0.4

    event_bus = DummyEventBus()
    app_state = AppState[DummyEventBus](
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
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
