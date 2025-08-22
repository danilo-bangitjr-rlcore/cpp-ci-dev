import logging
import random

import numpy as np
from lib_config.loader import load_config

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.offline.utils import (
    OfflineTraining,
    load_offline_transitions,
    offline_rl_from_buffer,
    run_offline_evaluation_phase,
)
from corerl.state import AppState

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@load_config(MainConfig)
def main(cfg: MainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    save_path = cfg.save_path

    # set the random seeds
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)

    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=DummyEventBus(),
    )

    pipeline = Pipeline(app_state, cfg.pipeline)
    assert isinstance(cfg.env, AsyncEnvConfig)
    data_reader = DataReader(db_cfg=cfg.env.db)

    column_desc = pipeline.column_descriptions
    agent = GreedyAC(cfg.agent, app_state, column_desc)

    # Offline training
    assert cfg.offline.offline_steps > 0
    offline_training = OfflineTraining(cfg)
    pipeline_out = load_offline_transitions(app_state, pipeline, data_reader)
    agent.update_buffer(pipeline_out)
    offline_rl_from_buffer(agent, cfg.offline.offline_steps)
    run_offline_evaluation_phase(cfg, app_state, agent, pipeline)

    app_state.metrics.close()
    app_state.evals.close()
    agent.close()
    agent.save(save_path / 'agent')


if __name__ == "__main__":
    main()
