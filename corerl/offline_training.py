import logging
import random

import numpy as np
import torch

import corerl.main_utils as utils
from corerl.agent.factory import init_agent
from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals import evals_group
from corerl.eval.metrics import metrics_group
from corerl.messages.event_bus import EventBus
from corerl.offline.utils import OfflineTraining
from corerl.state import AppState
from corerl.utils.device import device

log = logging.getLogger(__name__)


@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    save_path = utils.prepare_save_dir(cfg)
    device.update_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    app_state = AppState(
        metrics=metrics_group.dispatch(cfg.metrics),
        evals=evals_group.dispatch(cfg.evals),
        event_bus=EventBus(cfg.event_bus, cfg.env),
    )

    pipeline = Pipeline(cfg.pipeline)
    column_desc = pipeline.column_descriptions
    agent = init_agent(cfg.agent, app_state, column_desc)

    # Offline training
    assert cfg.experiment.offline_steps > 0
    offline_training = OfflineTraining(cfg)
    offline_training.load_offline_transitions(pipeline)
    offline_training.train(app_state, agent)

    app_state.metrics.close()
    app_state.evals.close()
    agent.close()
    agent.save(save_path / 'agent')


if __name__ == "__main__":
    main()
