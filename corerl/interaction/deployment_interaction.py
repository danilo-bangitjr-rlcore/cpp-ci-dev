import logging
from datetime import UTC, datetime, timedelta
from time import sleep
from typing import Generator

import numpy as np

from corerl.agent.base import BaseAgent
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.interaction.interaction import Interaction

logger = logging.getLogger(__file__)

@config()
class DepInteractionConfig:
    name: str = "dep_interaction"

class DeploymentInteraction(Interaction):
    def __init__(
        self,
        cfg: DepInteractionConfig,
        agent: BaseAgent,
        env: AsyncEnv,
        pipeline: Pipeline,
    ):
        self._pipeline = pipeline
        self._env = env
        self._agent = agent

        self._column_desc = pipeline.column_descriptions

        self._should_reset = True
        self._last_state: np.ndarray | None = None
        self._pipeline.register_hook(StageCode.SC, self._capture_last_state)

        ### timing logic ###
        self.obs_period = env.obs_period
        self.action_period = env.action_period
        self._next_action_timestamp: datetime = datetime.now(UTC) # take an action right away

        # the step clock starts ticking on the first invocation of `next(self._step_clock)`
        # this should occur on the first call to `self.step`
        self._step_clock = clock_generator(tick_period=self.obs_period)

    def step(self):
        step_timestamp = next(self._step_clock)
        wait_for_timestamp(step_timestamp)
        logger.info("beginning step logic")

        o = self._env.get_latest_obs()
        pr = self._pipeline(o, caller_code=CallerCode.ONLINE, reset_temporal_state=self._should_reset)
        if pr.transitions is not None:
            self._agent.update_buffer(pr.transitions)
        self._agent.update()

        s = self._get_latest_state()
        assert s is not None
        if self._should_take_action(step_timestamp):
            a = self._agent.get_action(s)
            self._env.emit_action(a)



    # ---------
    # internals
    # ---------
    def _should_take_action(self, step_timestamp: datetime) -> bool:
        if step_timestamp >= self._next_action_timestamp:
            self._next_action_timestamp = step_timestamp + self.action_period
            return True

        return False

    def _capture_last_state(self, pf: PipelineFrame):
        if pf.caller_code != CallerCode.ONLINE:
            return

        row = pf.data.tail(1)

        tags = self._column_desc.state_cols
        state = row[list(tags)].iloc[0].to_numpy()

        self._last_state = np.asarray(state, dtype=np.float32)
        logger.info(f"captured state {self._last_state}, with columns {tags}")


    def _get_latest_state(self) -> np.ndarray | None:
        if self._last_state is None:
            logger.error("Tried to get interaction state, but none existed")
            return None

        return self._last_state

def clock_generator(tick_period: timedelta) -> Generator[datetime, None, None]:
    tick = datetime.now(UTC)
    tick.replace(microsecond=0) # trim microseconds
    while True:
        yield tick
        tick += tick_period

def wait_for_timestamp(timestamp: datetime) -> None:
    """
    Blocks until the requested timestamp
    """
    now = datetime.now(UTC)
    if now >= timestamp:
        sleep_duration = 0
    else:
        sleep_duration = (timestamp - now).total_seconds()
    sleep(sleep_duration)

