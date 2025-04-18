#!/usr/bin/env python3

import logging
import random
import sys
import time
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.environment.async_env.factory import init_async_env
from corerl.eval.config import register_pipeline_evals
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.interaction.factory import init_interaction
from corerl.messages.event_bus import DummyEventBus, EventBus
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


def main_loop(cfg: MainConfig, app_state: AppState, pipeline: Pipeline, env: DeploymentAsyncEnv):
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

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(total=max_steps, disable=not cfg.experiment.is_simulation)

    app_state.event_bus.start()
    event_stream = app_state.event_bus.listen_forever()

    for event in event_stream:
        if event and event.type == EventType.step_get_obs:
            pbar.update(1)

        if cfg.experiment.is_simulation:
            interaction.step()
            pbar.update(1)

        if max_steps is not None and app_state.agent_step >= max_steps:
            break


def retryable_main(cfg: MainConfig):
    if cfg.log_path is not None:
        enable_log_files(cfg.log_path)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    torch.set_num_threads(cfg.experiment.num_threads)
    device.update_device(cfg.experiment.device)

    # build global objects
    event_bus = (
        EventBus(cfg.event_bus, cfg.env)
        if not cfg.experiment.is_simulation else
        DummyEventBus()
    )
    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=event_bus,
    )
    pipeline = Pipeline(app_state, cfg.pipeline)
    env = init_async_env(cfg.env, cfg.pipeline.tags)

    try:
        main_loop(cfg, app_state, pipeline, env)

    except Exception as e:
        log.exception(e)

    finally:
        app_state.metrics.close()
        app_state.evals.close()
        env.cleanup()
        event_bus.cleanup()


@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    # only do retry logic if we want to "run forever"
    if cfg.experiment.is_simulation:
        return retryable_main(cfg)

    # retry logic
    retries = 0
    last_error = datetime.now(UTC)

    while True:
        retryable_main(cfg)

        now = datetime.now(UTC)
        if now - last_error < timedelta(hours=1):
            # if less than an hour, retry up to 5 times
            # then exit if still failing
            last_error = now
            retries += 1
            if retries >= 5:
                log.error("Too many retries, exiting!")
                # exit code 70 corresponds to EX_SOFTWARE, which is unix-only
                sys.exit(70)

            # backoff exponentially over minutes
            min = 2 ** (retries - 1)
            time.sleep(60 * min)

        else:
            # if it has been more than an hour since the last error,
            # then perform the first retry
            retries = 1
            last_error = now


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
