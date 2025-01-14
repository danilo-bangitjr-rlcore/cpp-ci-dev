#!/usr/bin/env python3

import logging
import random
import sys
import os

import numpy as np
import torch

from corerl.agent.factory import init_agent
from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.factory import init_async_env
from corerl.environment.registry import register_custom_envs
from corerl.interaction.factory import init_interaction
from corerl.messages.client import make_msg_bus_client
from corerl.state import AppState, MetricsWriter
from corerl.utils.device import device

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    encoding="utf-8",
    level=logging.INFO,
)

@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    device.update_device(cfg.experiment.device)

    event_bus = make_msg_bus_client(cfg.agent.message_bus)
    event_bus.start_sync()

    app_state = AppState(
        metrics=MetricsWriter(cfg.metrics),
        event_bus=event_bus,
    )

    # get custom gym environments
    register_custom_envs()

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    pipeline = Pipeline(cfg.pipeline)

    env = init_async_env(cfg.env, cfg.pipeline.tags)
    try:
        state_dim, action_dim = pipeline.get_state_action_dims()
        agent = init_agent(
            cfg.agent,
            app_state,
            state_dim,
            action_dim,
        )
        interaction = init_interaction(
            cfg=cfg.interaction, agent=agent, env=env, pipeline=pipeline, tag_configs=cfg.pipeline.tags
        )

        steps = 0
        max_steps = cfg.experiment.max_steps
        run_forever = cfg.experiment.run_forever

        while True:
            log.info(f"taking step {steps}")
            interaction.step()
            steps += 1
            if steps >= max_steps and not run_forever:
                break


    except Exception as e:
        log.exception(e)
        sys.exit(os.EX_SOFTWARE)

    finally:
        env.cleanup()

if __name__ == "__main__":
    main()
