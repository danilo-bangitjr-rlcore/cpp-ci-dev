import logging
import random
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
from lib_defs.type_defs.base_events import EventTopic, EventType
from service_framework.service import RLTuneService

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.environment.async_env.factory import init_async_env
from corerl.eval.config import register_pipeline_evals
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.interaction.deployment_interaction import DeploymentInteraction
from corerl.interaction.factory import init_interaction
from corerl.messages.event_bus import DummyEventBus, EventBus
from corerl.state import AppState
from corerl.utils.app_time import AppTime

logger = logging.getLogger(__name__)


class CoreRLService(RLTuneService):
    def __init__(self, cfg: MainConfig):
        super().__init__(
            service_name='corerl',
            event_topic=EventTopic.corerl,
            event_bus_host=cfg.event_bus_client.host,
            event_bus_port=cfg.event_bus_client.port,
            event_bus_enabled=cfg.event_bus_client.enabled,
        )
        self.cfg = cfg

        self.app_state: AppState | None = None
        self.interaction: DeploymentInteraction | None = None
        self.env: DeploymentAsyncEnv | None = None
        self.pipeline: Pipeline | None = None
        self.step_count = 0

    async def _do_start(self):
        if self.cfg.log_path is not None:
            self._enable_log_files(self.cfg.log_path)

        seed = self.cfg.seed
        np.random.seed(seed)
        random.seed(seed)

        is_demo = self.cfg.demo_mode
        app_time = AppTime(
            is_demo=is_demo,
            start_time=datetime.now(UTC),
            obs_period=self.cfg.interaction.obs_period if is_demo else None,
        )

        if self.cfg.is_simulation:
            event_bus = DummyEventBus()
            self.app_state = AppState[DummyEventBus, MainConfig](
                cfg=self.cfg,
                metrics=create_metrics_writer(self.cfg.metrics, app_time.get_current_time),
                evals=create_evals_writer(self.cfg.evals),
                event_bus=event_bus,
                app_time=app_time,
            )
        else:
            event_bus = EventBus(self.cfg.event_bus)
            self.app_state = AppState[EventBus, MainConfig](
                cfg=self.cfg,
                metrics=create_metrics_writer(self.cfg.metrics, app_time.get_current_time),
                evals=create_evals_writer(self.cfg.evals),
                event_bus=event_bus,
                app_time=app_time,
            )

        self.pipeline = Pipeline(self.app_state, self.cfg.pipeline)
        self.env = init_async_env(self.cfg.env, self.cfg.pipeline.tags)

        column_desc = self.pipeline.column_descriptions
        agent = GreedyAC(
            self.cfg.agent,
            self.app_state,
            column_desc,
        )

        app_state = self.app_state
        assert app_state is not None
        assert self.env is not None
        assert self.pipeline is not None
        event_bus.attach_callback(event_type=EventType.flush_buffers, cb=lambda _e: app_state.evals.flush())
        register_pipeline_evals(self.cfg.eval_cfgs, agent, self.pipeline, self.app_state)

        self.interaction = init_interaction(
            cfg=self.cfg.interaction,
            app_state=self.app_state,
            agent=agent,
            env=self.env,
            pipeline=self.pipeline,
        )

        self.app_state.event_bus.start()

    async def _do_run(self) -> AsyncGenerator[None]:
        assert self.app_state is not None
        assert self.interaction is not None

        max_steps = self.cfg.max_steps
        event_stream = self.interaction.interact_forever()

        for self.step_count, _ in enumerate(event_stream):
            if max_steps is not None and self.app_state.agent_step >= max_steps:
                break

            event_bus = self.get_event_bus_client()
            if self.step_count % 100 == 0 and event_bus is not None:
                event_bus.emit_event(
                    EventType.step_agent_update,
                    topic=EventTopic.corerl,
                )

            yield

    async def _do_stop(self):
        if self.app_state is not None:
            self.app_state.stop_event.set()
            self.app_state.metrics.close()
            self.app_state.evals.close()
            self.app_state.event_bus.cleanup()

        if self.env is not None:
            self.env.cleanup()

        if self.interaction is not None:
            self.interaction.close()

    def _enable_log_files(self, log_path: Path):
        save_path = log_path / str(datetime.now(UTC).date())
        save_path.mkdir(exist_ok=True, parents=True)
        file_handler = RotatingFileHandler(
            filename=save_path / "rlcore.log",
            maxBytes=10_000_000,
            backupCount=3,
        )
        log_fmt = "[%(asctime)s][%(levelname)s] - %(message)s"
        file_handler.setFormatter(logging.Formatter(log_fmt))
        logging.getLogger().addHandler(file_handler)
