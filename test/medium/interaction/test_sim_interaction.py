import pytest

from datetime import timedelta
from sqlalchemy import Engine

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.factory import init_async_env
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.interaction.factory import init_interaction
from corerl.interaction.sim_interaction import SimInteraction
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState
from corerl.utils.time import percent_time_elapsed

@pytest.fixture()
def test_action_bounds(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    NUM_STEPS = 5
    cfg = direct_load_config(MainConfig, base='test/medium/interaction/assets', config_name='saturation.yaml')
    assert isinstance(cfg, MainConfig)
    cfg.env.db.port = tsdb_engine.url.port
    cfg.env.db.db_name = tsdb_tmp_db_name

    # build global objects
    event_bus = DummyEventBus()
    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=event_bus,
    )
    pipeline = Pipeline(app_state, cfg.pipeline)
    env = init_async_env(cfg.env, cfg.pipeline.tags)

    column_desc = pipeline.column_descriptions
    agent = GreedyAC(
        cfg.agent,
        app_state,
        column_desc,
    )

    interaction = init_interaction(
        cfg=cfg.interaction, app_state=app_state, agent=agent, env=env, pipeline=pipeline,
    )

    assert isinstance(interaction, SimInteraction)

    for _ in range(NUM_STEPS+1):
        interaction._on_get_obs()
        interaction._on_update()

        for _ in range(100):
            interaction._on_emit_action()
            assert interaction._last_action_df is not None
            last_a = interaction._last_action_df.to_numpy().item()
            p = percent_time_elapsed(
                start=app_state.start_time,
                end=app_state.start_time + timedelta(minutes=5),
            )

            low = (1-p)*0.25 + p*0
            high = (1-p)*0.26 + p*0.5
            assert low <= last_a <= high
