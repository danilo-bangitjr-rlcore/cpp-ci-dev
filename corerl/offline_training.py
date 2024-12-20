import logging
import numpy as np
import torch
import random

from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.utils.device import device
from corerl.agent.factory import init_agent
from corerl.offline.utils import load_offline_transitions, offline_training
from corerl.data_pipeline.pipeline import Pipeline

import corerl.utils.freezer as fr
import corerl.main_utils as utils

log = logging.getLogger(__name__)


@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    save_path = utils.prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    test_epochs = cfg.experiment.test_epochs
    device.update_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    pipeline = Pipeline(cfg.pipeline)
    state_dim, action_dim = pipeline.get_state_action_dims()
    agent = init_agent(cfg.agent, state_dim, action_dim)
    transitions = []

    # Offline training
    should_train_offline = cfg.experiment.offline_steps > 0
    if should_train_offline:
        offline_transitions = load_offline_transitions(cfg, pipeline)
        transitions += offline_transitions
        offline_training(cfg, agent, transitions)

    agent.close()
    agent.save(save_path / 'agent')


if __name__ == "__main__":
    main()
