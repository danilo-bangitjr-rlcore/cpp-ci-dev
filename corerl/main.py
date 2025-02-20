#!/usr/bin/env python3

import logging
import os
import random
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.factory import init_async_env
from corerl.environment.registry import register_custom_envs
from corerl.eval.config import register_pipeline_evals
from corerl.eval.evals import evals_group
from corerl.eval.metrics import metrics_group
from corerl.interaction.factory import init_interaction
from corerl.messages.event_bus import EventBus
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.device import device

log = logging.getLogger(__name__)
log_fmt = "[%(asctime)s][%(levelname)s] - %(message)s"
logging.basicConfig(
    format=log_fmt,
    encoding="utf-8",
    level=logging.INFO,
)

logging.getLogger('asyncua').setLevel(logging.CRITICAL)


def main_loop(cfg: MainConfig):
    event_bus = EventBus(cfg.event_bus, cfg.env)
    app_state = AppState(
        cfg=cfg,
        metrics=metrics_group.dispatch(cfg.metrics),
        evals=evals_group.dispatch(cfg.evals),
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
        column_desc = pipeline.column_descriptions
        agent = GreedyAC(
            cfg.agent,
            app_state,
            column_desc,
        )

        register_pipeline_evals(cfg.eval_cfgs, agent, pipeline, app_state)

        interaction = init_interaction(
            cfg=cfg.interaction, app_state=app_state, agent=agent, env=env, pipeline=pipeline,
        )

        steps = 0
        max_steps = cfg.experiment.max_steps
        run_forever = cfg.experiment.run_forever
        disable_pbar = False
        if run_forever:
            max_steps = 0
            disable_pbar = True
        pbar = tqdm(total=max_steps, disable=disable_pbar)

        event_bus.start()

        while True:
            if event_bus.enabled():
                event = event_bus.recv_event()
                if not event:
                    continue
                interaction.step_event(event)
                if event.type == EventType.step_get_obs:
                    pbar.update(1)
                    steps += 1
            else:
                interaction.step()
                pbar.update(1)
                steps += 1

            if not run_forever and steps >= max_steps:
                break

    except BaseException as e:
        log.exception(e)
        sys.exit(os.EX_SOFTWARE)

    finally:
        app_state.metrics.close()
        app_state.evals.close()
        env.cleanup()
        event_bus.cleanup()


@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    if cfg.log_path is not None:
        enable_log_files(cfg.log_path)

    torch.set_num_threads(cfg.experiment.num_threads)
    device.update_device(cfg.experiment.device)

    main_loop(cfg)


def enable_log_files(log_path: Path):
    save_path = log_path / str(datetime.now(UTC).date())
    save_path.mkdir(exist_ok=True, parents=True)
    file_handler = RotatingFileHandler(
        filename=save_path / "rlcore.log",
        maxBytes=10_000_000,
        backupCount=3,  # rotate over 3 log files
    )
    file_handler.setFormatter(logging.Formatter(log_fmt))
    logging.getLogger().addHandler(file_handler)


if __name__ == "__main__":
    main()
