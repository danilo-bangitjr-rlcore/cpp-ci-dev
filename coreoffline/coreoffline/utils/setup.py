import logging
import random

import numpy as np
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState

from coreoffline.utils.config import OfflineMainConfig


def create_standard_setup(cfg: OfflineMainConfig) -> tuple[AppState, Pipeline]:
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Set random seeds
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Create app state and pipeline
    app_state = AppState(
        cfg=cfg,
        metrics=create_metrics_writer(cfg.metrics),
        evals=create_evals_writer(cfg.evals),
        event_bus=DummyEventBus(),
    )
    pipeline = Pipeline(app_state, cfg.pipeline)

    return app_state, pipeline
