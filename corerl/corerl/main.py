#!/usr/bin/env python3

import logging
import random
import sys
import time
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
from lib_config.loader import load_config
from lib_defs.type_defs.base_events import EventTopic, EventType

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.factory import init_async_env
from corerl.eval.config import register_pipeline_evals
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.event_bus.client import EventBusClient
from corerl.interaction.deployment_interaction import DeploymentInteraction
from corerl.interaction.factory import init_interaction
from corerl.messages.event_bus import DummyEventBus, EventBus
from corerl.state import AppState
from corerl.tags.validate_tag_configs import validate_tag_configs
from corerl.utils.app_time import AppTime

log = logging.getLogger(__name__)
log_fmt = "[%(asctime)s][%(levelname)s] - %(message)s"
logging.getLogger('asyncua').setLevel(logging.CRITICAL)


def main_loop(
    cfg: MainConfig,
    app_state: AppState,
    interaction: DeploymentInteraction,
    event_bus_client: EventBusClient | None = None,
):
    max_steps = cfg.max_steps

    # event bus owns orchestration of interactions
    # driving loop below gives access to event stream
    app_state.event_bus.start()
    event_stream = interaction.interact_forever()

    for step_count, _ in enumerate(event_stream):
        if max_steps is not None and app_state.agent_step >= max_steps:
            break

        if event_bus_client is not None and step_count % 100 == 0:
            event_bus_client.emit_event(
                EventType.step_agent_update,
                topic=EventTopic.corerl,
            )


def retryable_main(cfg: MainConfig):
    if cfg.log_path is not None:
        enable_log_files(cfg.log_path)

    # set the random seeds
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)

    # initialize event bus client if enabled
    event_bus_client: EventBusClient | None = None
    if cfg.event_bus_client.enabled:
        event_bus_client = EventBusClient(
            host=cfg.event_bus_client.host,
            pub_port=cfg.event_bus_client.pub_port,
            sub_port=cfg.event_bus_client.sub_port,
        )
        event_bus_client.connect()
        log.info(
            f"Event bus client connected to "
            f"{cfg.event_bus_client.host}:{cfg.event_bus_client.pub_port}/{cfg.event_bus_client.sub_port}",
        )

    # build global objects
    is_demo = cfg.demo_mode
    app_time = AppTime(
        is_demo=is_demo,
        start_time=datetime.now(UTC),
        obs_period=cfg.interaction.obs_period if is_demo else None,
    )

    if cfg.is_simulation:
        event_bus = DummyEventBus()
        app_state = AppState[DummyEventBus, MainConfig](
            cfg=cfg,
            metrics=create_metrics_writer(cfg.metrics, app_time.get_current_time),
            evals=create_evals_writer(cfg.evals),
            event_bus=event_bus,
            app_time=app_time,
        )
    else:
        event_bus = EventBus(cfg.event_bus)
        app_state = AppState[EventBus, MainConfig](
            cfg=cfg,
            metrics=create_metrics_writer(cfg.metrics, app_time.get_current_time),
            evals=create_evals_writer(cfg.evals),
            event_bus=event_bus,
            app_time=app_time,
        )

    pipeline = Pipeline(app_state, cfg.pipeline)
    env = init_async_env(cfg.env, cfg.pipeline.tags)

    column_desc = pipeline.column_descriptions
    agent = GreedyAC(
        cfg.agent,
        app_state,
        column_desc,
    )

    event_bus.attach_callback(event_type=EventType.flush_buffers, cb=lambda _e: app_state.evals.flush())
    register_pipeline_evals(cfg.eval_cfgs, agent, pipeline, app_state)

    interaction = init_interaction(
        cfg=cfg.interaction,
        app_state=app_state,
        agent=agent, env=env,
        pipeline=pipeline,
    )

    if event_bus_client is not None:
        event_bus_client.emit_event(
            EventType.service_started,
            topic=EventTopic.corerl,
        )

    try:
        main_loop(cfg, app_state, interaction, event_bus_client)

    except Exception as e:
        log.exception(e)

        if event_bus_client is not None:
            event_bus_client.emit_event(
                EventType.service_error,
                topic=EventTopic.corerl,
            )

        # if we are in a simulation, then we want to forward
        # exceptions to fail the process
        if cfg.is_simulation:
            raise e

    finally:
        if event_bus_client is not None:
            event_bus_client.emit_event(
                EventType.service_stopped,
                topic=EventTopic.corerl,
            )

        app_state.stop_event.set()
        app_state.metrics.close()
        app_state.evals.close()
        env.cleanup()
        event_bus.cleanup()
        interaction.close()

        if event_bus_client is not None:
            event_bus_client.close()


@load_config(MainConfig)
def main(cfg: MainConfig):
    validate_tag_configs(cfg)
    logging.basicConfig(
        format=log_fmt,
        encoding="utf-8",
        level=logging.INFO if not cfg.silent else logging.WARN,
    )

    # only do retry logic if we want to "run forever"
    if cfg.is_simulation or cfg.max_steps is not None:
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
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"FATAL ERROR during corerl startup: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
