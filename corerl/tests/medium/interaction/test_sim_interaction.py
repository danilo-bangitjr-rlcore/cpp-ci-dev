from datetime import timedelta
from typing import Any

import pytest
from lib_agent.gamma_schedule import create_gamma_scheduler
from sqlalchemy import Engine

from corerl.agent.greedy_ac import GreedyAC
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.factory import init_async_env
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.interaction.factory import init_interaction
from corerl.interaction.sim_interaction import SimInteraction
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState
from corerl.utils.time import percent_time_elapsed
from tests.infrastructure.config import create_config_with_overrides


@pytest.fixture
def saturation_app_state(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    assert tsdb_engine.url.port is not None

    cfg = create_config_with_overrides(
        base_config_path='tests/medium/interaction/assets/saturation.yaml',
        overrides={
            'env.db.port': tsdb_engine.url.port,
            'env.db.db_name': tsdb_tmp_db_name,
        },
    )

    event_bus = DummyEventBus()
    return AppState(
        cfg=cfg,
        metrics=create_metrics_writer(cfg.metrics),
        evals=create_evals_writer(cfg.evals),
        event_bus=event_bus,
    )


def test_action_bounds(saturation_app_state: AppState):
    """
    Verify action bounds change correctly over time during simulation interaction.
    """
    NUM_STEPS = 3
    app_state = saturation_app_state
    cfg = app_state.cfg

    pipeline = Pipeline(app_state, cfg.pipeline)
    env = init_async_env(cfg.env, cfg.pipeline.tags)

    gamma_scheduler = create_gamma_scheduler(cfg.agent.gamma_schedule)
    agent = GreedyAC(
        cfg.agent,
        app_state,
        pipeline.column_descriptions,
        gamma_scheduler,
    )

    interaction = init_interaction(
        cfg=cfg.interaction, app_state=app_state, agent=agent, env=env, pipeline=pipeline,
    )

    assert isinstance(interaction, SimInteraction)

    for _ in range(NUM_STEPS+1):
        interaction._on_get_obs()
        interaction._on_update()

        for _ in range(10):
            interaction._on_emit_action()
            assert interaction._last_action_df is not None
            last_a: Any = interaction._last_action_df.to_numpy().item()
            p = percent_time_elapsed(
                start=app_state.start_time,
                end=app_state.start_time + timedelta(minutes=5),
            )

            low = (1-p)*0.25 + p*0
            high = (1-p)*0.26 + p*0.5
            assert low <= last_a <= high

    agent.close()
    env.close()
    app_state.metrics.close()
    app_state.evals.close()
