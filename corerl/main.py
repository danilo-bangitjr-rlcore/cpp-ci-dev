#!/usr/bin/env python3

import logging
import random

import numpy as np
import torch
from tqdm import tqdm

import corerl.main_utils as utils
import corerl.utils.freezer as fr
from corerl.agent.factory import init_agent
from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.factory import init_async_env
from corerl.environment.registry import register_custom_envs
from corerl.interaction.sim_interaction import SimInteraction
from corerl.utils.device import device

log = logging.getLogger(__name__)


@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    save_path = utils.prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    device.update_device(cfg.experiment.device)

    # get custom gym environments
    register_custom_envs()

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    pipeline = Pipeline(cfg.pipeline)

    env = init_async_env(cfg.env, cfg.pipeline.tags)

    state_dim, action_dim = pipeline.get_state_action_dims()
    agent = init_agent(
        cfg.agent,
        state_dim,
        action_dim,
    )

    interaction = SimInteraction(
        agent,
        env,
        pipeline,
        cfg.pipeline.tags,
    )

    for _ in tqdm(range(cfg.experiment.max_steps)):
        interaction.step()

if __name__ == "__main__":
    main()
