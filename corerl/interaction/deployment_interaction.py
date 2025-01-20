import logging
from datetime import UTC, datetime, timedelta
from time import sleep
from typing import Generator

import numpy as np
import pandas as pd

from corerl.agent.base import BaseAgent
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.interaction.interaction import Interaction
from corerl.state import AppState

logger = logging.getLogger(__file__)

@config()
class DepInteractionConfig:
    name: str = "dep_interaction"

class DeploymentInteraction(Interaction):
    def __init__(
        self,
        cfg: DepInteractionConfig,
        app_state: AppState,
        agent: BaseAgent,
        env: AsyncEnv,
        pipeline: Pipeline,
    ):
        self._app_state = app_state
        self._pipeline = pipeline
        self._env = env
        self._agent = agent

        self._column_desc = pipeline.column_descriptions

        self._should_reset = False
        self._last_state = np.full(self._column_desc.state_dim, np.nan)
        self._pipeline.register_hook(StageCode.SC, self._capture_last_state)

        ### timing logic ###
        self.obs_period = env.obs_period
        self.action_period = env.action_period
        self._next_action_timestamp = datetime.now(UTC) # take an action right away
        self._last_state_timestamp: datetime | None = None
        self._state_age_tol = env.action_tolerance

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
        if s is not None and self._should_take_action(step_timestamp):
            a = self._agent.get_action(s)
            self._env.emit_action(a)

        self._app_state.agent_step += 1


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

        state_df = pf.data.tail(1)
        state_timestamp = state_df.index[0]
        self._write_state_features(state_df)

        tags = self._column_desc.state_cols
        state = state_df[list(tags)].iloc[0].to_numpy()

        self._last_state = np.asarray(state, dtype=np.float32)
        if isinstance(state_timestamp, pd.Timestamp):
            self._last_state_timestamp = state_timestamp.to_pydatetime()
        else:
            self._last_state_timestamp = datetime.now(UTC)
        logger.info(f"captured state {self._last_state}, with columns {tags}")


    def _get_latest_state(self) -> np.ndarray | None:
        now = datetime.now(UTC)
        if self._last_state_timestamp is None:
            logger.error("Tried to get interaction state, but no state has been captured")
            return None

        if np.any(np.isnan(self._last_state)):
            logger.error("Tried to get interaction state, but there were nan values")
            return None

        if now - self._last_state_timestamp > self._state_age_tol:
            logger.error("Got a stale interaction state")
            return None

        return self._last_state

    def _write_state_features(self, state_df: pd.DataFrame) -> None:
        if len(state_df) != 1:
            logger.error(f"unexpected state df length: {len(state_df)}")

        for feat_name in state_df.columns:
            val = state_df[feat_name].values[0]
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=feat_name,
                value=val,
            )


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
