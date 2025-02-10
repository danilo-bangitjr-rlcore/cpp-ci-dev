import logging
import shutil
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from corerl.agent.base import BaseAgent
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline, PipelineReturn
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.eval.monte_carlo import MonteCarloEvaluator
from corerl.interaction.configs import DepInteractionConfig
from corerl.interaction.interaction import Interaction
from corerl.messages.events import Event, EventType
from corerl.messages.heartbeat import Heartbeat
from corerl.state import AppState
from corerl.utils.maybe import Maybe
from corerl.utils.time import clock_generator, split_into_chunks, wait_for_timestamp

logger = logging.getLogger(__name__)


class DeploymentInteraction(Interaction):
    def __init__(
        self,
        cfg: DepInteractionConfig,
        app_state: AppState,
        agent: BaseAgent,
        env: AsyncEnv,
        pipeline: Pipeline,
    ):
        assert isinstance(env, DeploymentAsyncEnv)
        self._cfg = cfg

        self._heartbeat = Heartbeat(cfg.heartbeat, env.get_cfg())
        self._app_state = app_state
        self._pipeline = pipeline
        self._env = env
        self._agent = agent

        self._column_desc = pipeline.column_descriptions

        self._last_state = np.full(self._column_desc.state_dim, np.nan)
        self._last_action: np.ndarray | None = None
        self._last_action_df: pd.DataFrame | None = None # used to ping setpoints

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
        chunk_start = Maybe(cfg.hist_chunk_start).or_else(time_stats.start).astimezone(UTC)
        chunk_end = time_stats.end
        logger.info(f"Offline chunks will be loaded from {chunk_start} to {chunk_end}")
        self._chunks = split_into_chunks(
            chunk_start,
            chunk_end,
            width=self.obs_period * cfg.historical_batch_size
        )

        # warmup pipeline
        self.warmup_pipeline()
        # checkpointing state
        self._last_checkpoint = datetime.now(UTC)
        if cfg.restore_checkpoint:
            self.restore_checkpoint()

        # evals
        self._monte_carlo_eval = MonteCarloEvaluator(app_state.cfg.eval_cfgs.monte_carlo, app_state, agent)


    # -----------------------
    # -- Lifecycle Methods --
    # -----------------------
    def _on_get_obs(self):
        self.load_historical_chunk()

        o = self._env.get_latest_obs()
        pipe_return = self._pipeline(o, data_mode=DataMode.ONLINE)
        self._log_rewards(pipe_return)
        self._agent.update_buffer(pipe_return)

        # perform evaluations
        self._monte_carlo_eval.execute(pipe_return)

        # capture latest state
        state = pipe_return.states
        self._last_state = (
            state
            .iloc[-1]
            .to_numpy(dtype=np.float32)
        )

        self._last_action = (
            pipe_return.actions
            .iloc[-1]
            .to_numpy(dtype=np.float32)
        )

        self._write_state_features(state)

        state_timestamp = state.index[-1]
        if isinstance(state_timestamp, pd.Timestamp):
            self._last_state_timestamp = state_timestamp.to_pydatetime()
        else:
            self._last_state_timestamp = datetime.now(UTC)

        tags = self._column_desc.state_cols
        logger.info(f"captured state {self._last_state}, with columns {tags}")

        self.maybe_checkpoint()
        self._app_state.agent_step += 1
        self._env.maybe_write_agent_step(step=self._app_state.agent_step)


    def _log_rewards(self, pipeline_return: PipelineReturn):
        # log rewards
        r = float(pipeline_return.rewards['reward'].iloc[0])
        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric='reward',
            value=r,
        )


    def _on_emit_action(self):
        sa = self._get_latest_state_action()

        now = datetime.now(UTC)
        if sa is None or not self._should_take_action(now):
            logger.warning(f'Tried to take action, however was unable: {sa}')
            return

        logger.info("Querying agent policy for new action")

        s, a = sa
        delta = self._agent.get_action(s)
        a_df = self._pipeline.action_constructor.assign_action_names(a, delta)
        a_df = self._pipeline.preprocessor.inverse(a_df)
        self._env.emit_action(a_df, log_action=True)
        self._last_action_df = a_df

    # ------------------
    # -- No Event Bus --
    # ------------------
    def step(self):
        step_timestamp = next(self._step_clock)
        wait_for_timestamp(step_timestamp)
        logger.info("Beginning step logic")
        self._heartbeat.healthcheck()

        self._on_get_obs()
        self._agent.update()
        self._on_emit_action()

    # ---------------
    # -- Event Bus --
    # ---------------
    def step_event(self, event: Event):
        logger.debug(f"Interaction received Event: {event}")
        self._heartbeat.healthcheck()
        match event.type:
            case EventType.step:
                self.step()

            case EventType.step_get_obs:
                self._on_get_obs()

            case EventType.step_agent_update:
                self._agent.update()

            case EventType.step_emit_action:
                self._on_emit_action()

            case EventType.ping_setpoints:
                if self._last_action_df is not None:
                    logger.debug("Pinging setpoints")
                    self._env.emit_action(self._last_action_df)

            case EventType.agent_step:
                self._app_state.agent_step += 1
                self._env.maybe_write_agent_step(step=self._app_state.agent_step)

            case _:
                logger.warning(f"Unexpected step_event: {event}")


    # ---------------------
    # -- Historical Data --
    # ---------------------
    def warmup_pipeline(self):
        if self._cfg.warmup_period is None:
            return

        warmup_end = datetime.now(UTC)
        warmup_obs = self._env.data_reader.batch_aggregated_read(
            names=self._env.tag_names,
            start_time=warmup_end - self._cfg.warmup_period,
            end_time=warmup_end,
            bucket_width=self.obs_period
        )
        self._pipeline(warmup_obs, data_mode=DataMode.ONLINE)

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
        logger.info(f"Loading chunk data from {chunk_data.index[0]} to {chunk_data.index[-1]}")

        pipeline_out = self._pipeline(
            data=chunk_data,
            data_mode=DataMode.OFFLINE,
            reset_temporal_state=False,
        )

        self._agent.update_buffer(pipeline_out)


    # -------------------
    # -- Checkpointing --
    # -------------------
    def maybe_checkpoint(self):
        now = datetime.now(UTC)
        if now - self._last_checkpoint > timedelta(hours=1):
            self.checkpoint()


    def checkpoint(self):
        now = datetime.now(UTC)
        path = self._cfg.checkpoint_path / f'{str(now).replace(':','_')}'
        path.mkdir(exist_ok=True, parents=True)
        self._agent.save(path)
        self._last_checkpoint = now

        chkpoints = self._cfg.checkpoint_path.glob('*')
        for chk in chkpoints:
            time = datetime.fromisoformat(chk.name.replace('_',':'))
            if now - time > timedelta(days=1):
                shutil.rmtree(chk)


    def restore_checkpoint(self):
        chkpoints = list(self._cfg.checkpoint_path.glob('*'))
        if len(chkpoints) == 0:
            return

        # get latest checkpoint
        checkpoint = sorted(chkpoints)[-1]
        logger.info(f"Loading agent weights from checkpoint {checkpoint}")
        self._agent.load(checkpoint)


    # ---------
    # internals
    # ---------
    def _should_take_action(self, step_timestamp: datetime) -> bool:
        if step_timestamp >= self._next_action_timestamp:
            self._next_action_timestamp = step_timestamp + self.action_period
            return True

        return False


    def _get_latest_state_action(self) -> tuple[np.ndarray, np.ndarray] | None:
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

        if self._last_action is None:
            logger.error('Got a valid state, but had no prior action')
            return None

        return self._last_state, self._last_action

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
