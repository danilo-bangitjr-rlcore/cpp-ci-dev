import logging
import shutil
from datetime import UTC, datetime, timedelta
from typing import Any, Generator

import numpy as np
import pandas as pd

import corerl.eval.agent as agent_eval
from corerl.agent.greedy_ac import GreedyAC
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.eval.monte_carlo import MonteCarloEvaluator
from corerl.interaction.configs import InteractionConfig
from corerl.messages.events import Event, EventType
from corerl.messages.heartbeat import Heartbeat
from corerl.state import AppState
from corerl.utils.list import find
from corerl.utils.maybe import Maybe
from corerl.utils.time import clock_generator, percent_time_elapsed, split_into_chunks, wait_for_timestamp

logger = logging.getLogger(__name__)


class DeploymentInteraction:
    def __init__(
        self,
        cfg: InteractionConfig,
        app_state: AppState,
        agent: GreedyAC,
        env: DeploymentAsyncEnv,
        pipeline: Pipeline,
    ):

        ### Core config ###
        self._cfg = cfg
        self._app_state = app_state
        self._env = env
        self._pipeline = pipeline
        self._agent = agent

        ### State-Action Management ###
        self._column_desc = pipeline.column_descriptions
        self._last_state = np.full(self._column_desc.state_dim, np.nan)
        self._last_action = np.full(self._column_desc.action_dim, np.nan)
        self._last_action_df: pd.DataFrame | None = None # used to ping setpoints

        ### Timing logic ###
        self.obs_period = env.obs_period
        self.action_period = env.action_period
        self._step_clock = clock_generator(tick_period=self.obs_period)
        self._next_action_timestamp = datetime.now(UTC) # take an action right away
        self._last_state_timestamp: datetime | None = None
        self._state_age_tol = env.action_tolerance

        ### Evals ###
        self._monte_carlo_eval = MonteCarloEvaluator(
            app_state.cfg.eval_cfgs.monte_carlo,
            app_state,
            agent,
        )

        ### Heartbeat (to be replaced by coreio)###
        self._heartbeat = self._init_heartbeat()

        ### Offline data and pretraining ###
        self._first_online_timestamp = datetime.now(UTC) # used in background offline data loading
        self._offline_chunks = self._init_offline_chunks()
        self._pretrain()

        ### Checkpointing state ###
        self._last_checkpoint = datetime.now(UTC)
        if cfg.restore_checkpoint:
            self.restore_checkpoint()

        ### Warmup Pipeline ###
        self.warmup_pipeline()

    def _init_offline_chunks(self) -> Generator[tuple[datetime, datetime], Any, None] | None:
        if not self._cfg.load_historical_data:
            return None

        time_stats = self._env.data_reader.get_time_stats()
        chunk_start = Maybe(self._cfg.hist_chunk_start).or_else(time_stats.start).astimezone(UTC)
        chunk_end = time_stats.end
        logger.info(f"Offline chunks will be loaded from {chunk_start} to {chunk_end}")
        return split_into_chunks(
            chunk_start,
            chunk_end,
            width=self.obs_period * self._cfg.historical_batch_size
        )

    def _pretrain(self):
        # load first historical chunk
        self.load_historical_chunk()
        # and then perform warmup updates
        for _ in range(self._cfg.update_warmup):
            self._agent.update()

    def _init_heartbeat(self):
        ### Comms management to be replaced by interaction with coreio ###
        return Heartbeat(self._cfg.heartbeat, self._env.get_cfg().coreio_origin)


    # -----------------------
    # -- Lifecycle Methods --
    # -----------------------
    def _on_get_obs(self):
        self.load_historical_chunk()

        o = self._env.get_latest_obs()
        pipe_return = self._pipeline(o, data_mode=DataMode.ONLINE, reset_temporal_state=self._should_reset(o))
        self._agent.update_buffer(pipe_return)

        # capture latest state
        self._last_state = (
            pipe_return.states
            .iloc[-1]
            .to_numpy(dtype=np.float32)
        )

        self._last_action = (
            pipe_return.actions
            .iloc[-1]
            .to_numpy(dtype=np.float32)
        )

        # log states
        self._write_to_metrics(pipe_return.states, prefix='STATE-')

        # log rewards
        self._write_to_metrics(pipe_return.rewards) # no prefix required

        # perform evaluations
        self._monte_carlo_eval.execute(pipe_return, "online")

        state_timestamp = pipe_return.states.index[-1]
        if isinstance(state_timestamp, pd.Timestamp):
            self._last_state_timestamp = state_timestamp.to_pydatetime()
        else:
            self._last_state_timestamp = datetime.now(UTC)

        tags = self._column_desc.state_cols
        logger.info(f"captured state {self._last_state}, with columns {tags}")

        self.maybe_checkpoint()
        self._app_state.agent_step += 1


    def _on_emit_action(self):
        sa = self._get_latest_state_action()

        now = datetime.now(UTC)
        if sa is None or not self._should_take_action(now):
            logger.warning(f'Tried to take action, however was unable: {sa}')
            return

        logger.info("Querying agent policy for new action")

        s, prev_a = sa
        next_a = self._agent.get_action_interaction(s, prev_a)
        norm_next_a_df = self._pipeline.action_constructor.get_action_df(next_a)
        next_a_df = self._pipeline.preprocessor.inverse(norm_next_a_df)
        next_a_df = self._clip_action_bounds(next_a_df)
        self._env.emit_action(next_a_df, log_action=True)
        self._last_action_df = next_a_df

        # metrics + eval
        agent_eval.policy_variance(self._app_state, self._agent, s, prev_a)
        agent_eval.q_online(self._app_state, self._agent, s, next_a)
        agent_eval.greed_dist_online(self._app_state, self._agent, s, prev_a)
        agent_eval.greed_values_online(self._app_state, self._agent, s, prev_a)
        agent_eval.q_values_and_act_prob(self._app_state, self._agent, s, prev_a)

        # log actions
        self._write_to_metrics(next_a_df, prefix='ACTION-')

    def _on_update(self):
        self._agent.update()

        # metrics + eval
        agent_eval.greed_dist_batch(self._app_state, self._agent)
        agent_eval.greed_values_batch(self._app_state, self._agent)



    # ------------------
    # -- No Event Bus --
    # ------------------
    def step(self):
        self._wait_for_next_step()
        logger.info("Beginning step logic")
        self._heartbeat.healthcheck()

        self._on_get_obs()
        self._on_update()
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
                self._on_update()

            case EventType.step_emit_action:
                self._on_emit_action()

            case EventType.ping_setpoints:
                if self._last_action_df is not None:
                    logger.debug("Pinging setpoints")
                    self._env.emit_action(self._last_action_df)

            case EventType.agent_step:
                self._app_state.agent_step += 1

            case _:
                logger.warning(f"Unexpected step_event: {event}")

    # ---------
    # internals
    # ---------
    def _should_reset(self, observation: pd.DataFrame) -> bool:
        return False


    def _wait_for_next_step(self):
        next_step_timestamp = next(self._step_clock)
        wait_for_timestamp(next_step_timestamp)

    def _clip_action_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        for ai_sp_tag in df.columns:
            df = self._clip_single_action(df, ai_sp_tag)

        return df


    def _clip_single_action(self, df: pd.DataFrame, ai_sp_tag: str) -> pd.DataFrame:
        cfg = find(lambda cfg: cfg.name == ai_sp_tag, self._pipeline.tags)
        assert cfg is not None, f'Failed to find tag config for {ai_sp_tag}'
        assert cfg.operating_range is not None, 'AI setpoint tag must have an operating range'

        guard_lo, guard_hi = cfg.operating_range
        if cfg.guardrail_schedule is None:
            return df

        perc = self._get_elapsed_guardrail_duration(cfg.guardrail_schedule.duration)

        start_lo, start_hi = cfg.guardrail_schedule.starting_range

        if start_lo is not None:
            assert guard_lo is not None
            guard_lo = (1 - perc) * start_lo + perc * guard_lo

        if start_hi is not None:
            assert guard_hi is not None
            guard_hi = (1 - perc) * start_hi + perc * guard_hi

        df[ai_sp_tag] = df[ai_sp_tag].clip(lower=guard_lo, upper=guard_hi)
        return df

    def _get_elapsed_guardrail_duration(self, guardrail_duration: timedelta):
        return percent_time_elapsed(
            start=self._app_state.start_time,
            end=self._app_state.start_time + guardrail_duration,
        )


    def _should_take_action(self, curr_time: datetime) -> bool:
        if self._app_state.event_bus.enabled():
            return True
        if curr_time >= self._next_action_timestamp:
            self._next_action_timestamp = curr_time + self.action_period
            return True

        return False


    def _get_latest_state_action(self) -> tuple[np.ndarray, np.ndarray] | None:
        if (
            self._state_is_fresh() and
            self._state_has_no_nans() and
            self._action_has_no_nans()
        ):
            return self._last_state, self._last_action

        return None

    def _state_is_fresh(self):
        if self._last_state_timestamp is None:
            logger.error("Interaction state is None")
            return False
        if datetime.now(UTC) - self._last_state_timestamp > self._state_age_tol:
            logger.error("Interaction state is stale")
            return False
        return True

    def _state_has_no_nans(self):
        if np.any(np.isnan(self._last_state)):
            logger.error("Interaction state contains nan values")
            return False
        return True

    def _action_has_no_nans(self):
        if np.any(np.isnan(self._last_action)):
            logger.error("Last action contains nan values")
            return False
        return True


    def _write_to_metrics(self, df: pd.DataFrame, prefix: str = '') -> None:
        if len(df) != 1:
            logger.error(f"unexpected df length: {len(df)}")

        for feat_name in df.columns:
            val = df[feat_name].values[0]
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=prefix + feat_name,
                value=val,
            )


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
            bucket_width=self.obs_period,
            tag_aggregations=self._env.tag_aggs,
        )
        self._pipeline(warmup_obs, data_mode=DataMode.ONLINE)

    def load_historical_chunk(self):
        if self._offline_chunks is None:
            return
        try:
            start_time, end_time = next(self._offline_chunks)
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
            tag_aggregations=self._env.tag_aggs,
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
        self._app_state.save(path)
        self._last_checkpoint = now

        chkpoints = self._cfg.checkpoint_path.glob('*')
        for chk in chkpoints:
            time = datetime.fromisoformat(chk.name.replace('_',':'))
            if now - time > timedelta(days=1):
                shutil.rmtree(chk)


    def restore_checkpoint(self):
        if not self._cfg.restore_checkpoint:
            return

        chkpoints = list(self._cfg.checkpoint_path.glob('*'))
        if len(chkpoints) == 0:
            return

        # get latest checkpoint
        checkpoint = sorted(chkpoints)[-1]
        logger.info(f"Loading agent weights from checkpoint {checkpoint}")
        self._agent.load(checkpoint)
        self._app_state.load(checkpoint)


