import os

import jax.numpy as jnp
import pytest
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


@pytest.fixture
def nominal_setpoint_config():
    config_path = os.path.join(os.path.dirname(__file__), 'assets/nominal_setpoint.yaml')
    cfg = create_config_with_overrides(base_config_path=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    return cfg


@pytest.fixture
def nominal_setpoint_app_state(nominal_setpoint_config: MainConfig):
    event_bus = DummyEventBus()
    return AppState[DummyEventBus, MainConfig](
        cfg=nominal_setpoint_config,
        metrics=create_metrics_writer(nominal_setpoint_config.metrics),
        evals=create_evals_writer(nominal_setpoint_config.evals),
        event_bus=event_bus,
    )


@pytest.fixture
def nominal_setpoint_agent(nominal_setpoint_config: MainConfig, nominal_setpoint_app_state: AppState):
    pipeline = Pipeline(nominal_setpoint_app_state, nominal_setpoint_config.pipeline)
    return GreedyAC(
        nominal_setpoint_config.agent,
        nominal_setpoint_app_state,
        pipeline.column_descriptions,
    )


def test_nominal_setpoint_norm(
    nominal_setpoint_config: MainConfig,
    nominal_setpoint_agent: GreedyAC,
):
    """
    Verify that nominal_setpoint config flows through to agent._nominal_setpoints array.
    """
    assert isinstance(nominal_setpoint_config.pipeline.tags[0], SetpointTagConfig)
    assert nominal_setpoint_config.pipeline.tags[0].nominal_setpoint == 0.4
    assert jnp.isclose(nominal_setpoint_agent._nominal_setpoints, jnp.array([0.4]))
