#!/usr/bin/env python3

import logging
import os
import random
import sys
import threading

import numpy as np
import torch
import zmq
from tqdm import tqdm

from corerl.agent.factory import init_agent
from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.factory import init_async_env
from corerl.environment.registry import register_custom_envs
from corerl.eval.writer import metrics_group
from corerl.interaction.factory import init_interaction
from corerl.messages.events import Event, EventTopic
from corerl.messages.scheduler import scheduler_task
from corerl.state import AppState
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

    scheduler = None
    socket = None
    event_bus = None
    stop_event = None

    # Spin up scheduler process
    if cfg.event_bus.enabled:
        # scheduler = multiprocessing.Process(target=scheduler_task, args=(cfg.event_bus,))
        context = zmq.Context()
        stop_event = threading.Event()
        scheduler = threading.Thread(target=scheduler_task, args=(cfg.event_bus, context, stop_event))
        scheduler.setDaemon(True)
        scheduler.start()

        event_bus = context.socket(zmq.PUB)
        event_bus.bind(cfg.event_bus.app_connection)

        socket = context.socket(zmq.SUB)
        socket.connect(cfg.event_bus.scheduler_connection)
        socket.connect(cfg.event_bus.cli_connection)
        socket.connect(cfg.event_bus.app_connection)
        socket.setsockopt_string(zmq.SUBSCRIBE, EventTopic.corerl)  # Empty string ("") to subscribe to everything

    app_state = AppState(
        metrics=metrics_group.dispatch(cfg.metrics),
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
        agent = init_agent(
            cfg.agent,
            app_state,
            column_desc,
        )
        interaction = init_interaction(
            cfg=cfg.interaction, app_state=app_state, agent=agent, env=env, pipeline=pipeline,
        )

        steps = 0
        max_steps = cfg.experiment.max_steps
        run_forever = cfg.experiment.run_forever
        if run_forever:
            max_steps = 0
        pbar = tqdm(total=max_steps)

        while True:
            pbar.update(1)
            if socket:
                raw_payload = socket.recv()
                raw_topic, raw_event = raw_payload.split(b" ", 1)
                event = Event.model_validate_json(raw_event)
                interaction.step_event(event)
            else:
                interaction.step()
            steps += 1
            if not run_forever and steps >= max_steps:
                break

    except Exception as e:
        log.exception(e)
        sys.exit(os.EX_SOFTWARE)

    finally:
        app_state.metrics.close()
        env.cleanup()

        if stop_event:
            stop_event.set()

        if scheduler:
            scheduler.join()


if __name__ == "__main__":
    main()
