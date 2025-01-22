import logging
from datetime import UTC, datetime, timedelta
from time import sleep
from typing import Generator, Literal

import numpy as np
import pandas as pd

from corerl.agent.base import BaseAgent
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.environment.async_env.opc_tsdb_sim_async_env import OPCTSDBSimAsyncEnv
from corerl.interaction.interaction import Interaction
from corerl.state import AppState
from corerl.utils.time import split_into_chunks

logger = logging.getLogger(__file__)

@config()
class DepInteractionConfig:
    name: Literal["dep_interaction"] = "dep_interaction"
    historical_batch_size: int = 10000


class DeploymentInteraction(Interaction):
    def __init__(
        self,
        cfg: DepInteractionConfig,
        app_state: AppState,
        agent: BaseAgent,
        env: AsyncEnv,
        pipeline: Pipeline,
    ):
        assert isinstance(env, DeploymentAsyncEnv) or isinstance(env, OPCTSDBSimAsyncEnv)

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

        # stateful info for background offline data loading
        self._first_online_timestamp = datetime.now(UTC)

        time_stats = self._env.data_reader.get_time_stats()
        self._chunks = split_into_chunks(
            time_stats.start,
            time_stats.end,
            width=self.obs_period * cfg.historical_batch_size
        )

    def step(self):
        step_timestamp = next(self._step_clock)
        wait_for_timestamp(step_timestamp)
        logger.info("beginning step logic")

        self.load_historical_chunk()

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

    def load_historical_chunk(self):
        try:
            start_time, end_time = next(self._chunks)
        except StopIteration:
            return

        end_time = min(end_time, self._first_online_timestamp)
        if end_time - start_time < self.obs_period:
            return

        tag_names = [tag_cfg.name for tag_cfg in self._pipeline.tags]
        chunk_data = self._env.data_reader.batch_aggregated_read(
            names=tag_names,
            start_time=start_time,
            end_time=end_time,
            bucket_width=self.obs_period,
        )

        pipeline_out = self._pipeline(
            data=chunk_data,
            caller_code=CallerCode.OFFLINE,
            reset_temporal_state=False,
        )

        if pipeline_out.transitions is not None:
            self._agent.update_buffer(pipeline_out.transitions)


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
