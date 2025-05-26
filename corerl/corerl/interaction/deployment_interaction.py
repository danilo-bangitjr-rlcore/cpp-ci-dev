import functools
import logging
import math
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

import corerl.eval.agent as agent_eval
from corerl.agent.greedy_ac import GreedyAC
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline, PipelineReturn
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.eval.hindsight_return import HindsightReturnEval
from corerl.eval.monte_carlo import MonteCarloEvaluator
from corerl.interaction.configs import InteractionConfig
from corerl.messages.events import Event, EventType
from corerl.messages.heartbeat import Heartbeat
from corerl.messages.scheduler import start_scheduler_thread
from corerl.state import AppState
from corerl.utils.list import sort_by
from corerl.utils.time import clock_generator, split_windows_into_chunks

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    feats: np.ndarray
    action_lo: np.ndarray
    action_hi: np.ndarray
    timestamp: datetime | None = None

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
        self._last_state: AgentState = AgentState(
            feats=np.full(self._column_desc.state_dim, np.nan),
            action_lo=np.full(self._column_desc.action_dim, np.nan),
            action_hi=np.full(self._column_desc.action_dim, np.nan)
        )
        self._last_action_df: pd.DataFrame | None = None # used to ping setpoints

        ### Timing logic ###
        self.obs_period = cfg.obs_period
        self.action_period = cfg.action_period
        self._step_clock = clock_generator(tick_period=self.obs_period)
        self._next_action_timestamp = datetime.now(UTC) # take an action right away
        self._state_age_tol = cfg.state_age_tol

        ### Evals ###
        self._monte_carlo_eval = MonteCarloEvaluator(
            app_state.cfg.eval_cfgs.monte_carlo,
            app_state,
            agent,
        )

        self._hs_return_eval = HindsightReturnEval(
            app_state.cfg.eval_cfgs.avg_reward,
            app_state,
        )

        ### Heartbeat (to be replaced by coreio)###
        self._heartbeat = self._init_heartbeat()

        ### Offline data and pretraining ###
        self._first_online_timestamp = datetime.now(UTC) # used in background offline data loading
        self._offline_chunks = self._init_offline_chunks()
        self._pretrain()

        ### Checkpointing state ###
        self._checkpoint_freq = cfg.checkpoint_freq
        self._checkpoint_cliff = cfg.checkpoint_cliff
        self._last_checkpoint = datetime.now(UTC)
        if cfg.restore_checkpoint:
            self.restore_checkpoint()

        ### Warmup Pipeline ###
        self.warmup_pipeline()

        ### Lifecycle methods ###
        self._scheduler: threading.Thread | None = None
        self._app_state.event_bus.attach_callbacks({
            EventType.step_emit_action:     self._handle_event(self._on_emit_action),
            EventType.step_get_obs:         self._handle_event(self._on_get_obs),
            EventType.step_agent_update:    self._handle_event(self._on_update),
            EventType.ping_setpoints:       self._handle_event(self._on_ping_setpoint),
        })

    def _init_offline_chunks(self) -> Generator[tuple[datetime, datetime], Any, None] | None:
        if not self._cfg.load_historical_data:
            return None

        if len(self._cfg.historical_windows) == 0:
            # default to one window from beginning to end of recorded time
            self._cfg.historical_windows = [(None, None)]

        # impute first and/or last timestamp in sequence if necessary
        time_stats = self._env.data_reader.get_time_stats()
        first_window = self._cfg.historical_windows[0]
        last_window = self._cfg.historical_windows[-1]
        if first_window[0] is None:
            self._cfg.historical_windows[0] = (time_stats.start, first_window[1])
        if last_window[1] is None:
            self._cfg.historical_windows[-1] = (last_window[0], time_stats.end)

        return split_windows_into_chunks(
            windows=self._cfg.historical_windows,
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


    def interact_forever(self):
        self._scheduler = start_scheduler_thread(self._app_state)
        event_stream = self._app_state.event_bus.listen_forever()
        yield from event_stream

    # -----------------------
    # -- Lifecycle Methods --
    # -----------------------
    def _on_get_obs(self):
        self.load_historical_chunk()

        o = self._env.get_latest_obs()
        if self._cfg.write_obs_to_csv:
            self._write_obs_to_csv(o)
        pipe_return = self._pipeline(o, data_mode=DataMode.ONLINE, reset_temporal_state=self._should_reset(o))
        self._agent.update_buffer(pipe_return)
        self._capture_latest_state(pipe_return)

        # log states
        self._write_to_metrics(pipe_return.states, prefix='STATE-')

        # log rewards
        self._write_to_metrics(pipe_return.rewards) # no prefix required

        # perform evaluations
        self._hs_return_eval.execute(pipe_return.rewards)
        self._monte_carlo_eval.execute(pipe_return, "online")

        tags = self._column_desc.state_cols
        logger.info(f"captured state {self._last_state}, with columns {tags}")

        self.maybe_checkpoint()
        self._app_state.agent_step += 1


    def _on_emit_action(self):
        state = self._last_state
        next_a = self._get_action(state)
        if next_a is None:
            return

        norm_next_a_df = self._pipeline.action_constructor.get_action_df(next_a)
        # clip to the normalized action bounds
        norm_next_a_df = self._clip_action_bounds(norm_next_a_df, state.action_lo, state.action_hi)
        next_a_df = self._pipeline.preprocessor.inverse(norm_next_a_df)
        self._env.emit_action(next_a_df, log_action=True)
        self._last_action_df = next_a_df

        # log actions
        self._write_to_metrics(next_a_df, prefix='ACTION-')

        if not self._state_has_no_nans():
            return

        # eval
        agent_eval.q_online(self._app_state, self._agent, state.feats, next_a)
        agent_eval.greed_dist_online(self._app_state, self._agent, state.feats, state.action_lo, state.action_hi)
        agent_eval.greed_values_online(self._app_state, self._agent, state.feats, state.action_lo, state.action_hi)
        agent_eval.q_values_and_act_prob(self._app_state, self._agent, state.feats, state.action_lo, state.action_hi)



    def _on_update(self):
        self._agent.update()

        # metrics + eval
        agent_eval.greed_dist_batch(self._app_state, self._agent)
        agent_eval.greed_values_batch(self._app_state, self._agent)


    def _on_ping_setpoint(self):
        if self._last_action_df is None:
            return

        logger.debug("Pinging setpoints")
        self._env.emit_action(self._last_action_df)

    # ------------------
    # -- No Event Bus --
    # ------------------
    def step(self):
        self._on_get_obs()
        self._on_update()
        self._on_emit_action()

    # ---------------
    # -- Event Bus --
    # ---------------
    def _handle_event(self, f: Callable[[], None]):
        @functools.wraps(f)
        def _inner(event: Event):
            logger.debug(f"Interaction received Event: {event}")
            self._heartbeat.healthcheck()
            return f()

        return _inner

    def close(self):
        if self._scheduler is None:
            return

        self._scheduler.join(timeout=1)


    # ---------
    # internals
    # ---------
    def _get_action(self, state: AgentState) -> np.ndarray | None:
        if self._state_is_fresh() and self._state_has_no_nans():
            logger.info("Querying agent policy for new action")
            return self._agent.get_action_interaction(state.feats, state.action_lo, state.action_hi)

        else:
            self._app_state.event_bus.emit_event(EventType.action_period_reset)
            logger.warning(f'Tried to take action, however was unable: {state}')
            return None

    def _capture_latest_state(self, pipe_return: PipelineReturn):
        pr_ts = pipe_return.states.index[-1]
        self._last_state = AgentState(
            feats=pipe_return.states.iloc[-1].to_numpy(dtype=np.float32),
            action_lo=pipe_return.action_lo.iloc[-1].to_numpy(dtype=np.float32),
            action_hi=pipe_return.action_hi.iloc[-1].to_numpy(dtype=np.float32),
            timestamp=pr_ts.to_pydatetime() if isinstance(pr_ts, pd.Timestamp) else datetime.now(UTC)
        )


    def _clip_action_bounds(self, df: pd.DataFrame, action_lo : np.ndarray, action_hi: np.ndarray) -> pd.DataFrame:
        return df.clip(lower=action_lo, upper=action_hi)

    def _should_reset(self, observation: pd.DataFrame) -> bool:
        return False

    def _state_is_fresh(self):
        if self._last_state.timestamp is None:
            logger.error("Interaction state is None")
            return False
        if datetime.now(UTC) - self._last_state.timestamp > self._state_age_tol:
            logger.error("Interaction state is stale")
            return False
        return True

    def _state_has_no_nans(self):
        if np.any(np.isnan(self._last_state.feats)):
            logger.error("Interaction state features contains nan values")
            nan_tags = []
            for i, tag in enumerate(self._last_state.feats):
                if np.isnan(tag):
                    nan_tags.append(self._column_desc.state_cols[i])

            logger.error(f"nan tags: {nan_tags}")
            return False

        if np.any(np.isnan(self._last_state.action_lo)):
            logger.error("Action lower bound contains nan values")
            return False

        if np.any(np.isnan(self._last_state.action_hi)):
            logger.error("Action upper bound contains nan values")
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

    def _write_obs_to_csv(self, df: pd.DataFrame):
        csv_path = self._app_state.cfg.save_path / "observations.csv"
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', index=True, header=False)
        else:
            df.to_csv(csv_path, index=True)


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

        chunk_data = self._env.data_reader.batch_aggregated_read(
            names=self._env.tag_names,
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
        if now - self._last_checkpoint >= self._checkpoint_freq:
            self.checkpoint()


    def checkpoint(self):
        """
        Checkpoints and removes old checkpoints to maintain a set of checkpoints that get incresingly sparse with age.
        """
        now = datetime.now(UTC)
        path = self._cfg.checkpoint_path / f'{str(now).replace(':','_')}'
        path.mkdir(exist_ok=True, parents=True)
        self._agent.save(path)
        self._app_state.save(path)
        self._last_checkpoint = now

        chkpoints = list(self._cfg.checkpoint_path.glob('*'))
        times = [datetime.fromisoformat(chk.name.replace('_',':')) for chk in chkpoints]
        chkpoints, times = sort_by(chkpoints, times) # sorted oldest to youngest

        # keep all checkpoints more recent than the cliff
        cliff = now - self._checkpoint_cliff
        to_delete = prune_checkpoints(chkpoints, times, cliff, self._checkpoint_freq)

        for chk in to_delete:
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

def next_power_of_2(x: int):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

def prev_power_of_2(x: int):
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)

def periods_since(start: datetime, end: datetime, period: timedelta):
    return math.floor((end - start) / period)

def prune_checkpoints(
        chkpoints: list[Path],
        times: list[datetime],
        cliff: datetime,
        checkpoint_freq: timedelta
    ) -> list[Path]:

    to_delete = []
    for i, chk in enumerate(chkpoints):
        # keep latest and first checkpoint
        if i in (0, len(chkpoints) - 1):
            continue

        # keep all checkpoints more recent than the cliff
        if times[i] > cliff:
            continue

        periods_since_cliff_chk = periods_since(times[i], cliff, checkpoint_freq)
        periods_since_clif_prev_chk = periods_since(times[i-1], cliff, checkpoint_freq)
        periods_since_cliff_next_chk = periods_since(times[i+1], cliff, checkpoint_freq)

        # having checkpoints at powers of two is our goal. Get the next and previous powers of two in periods
        next_power_2 = next_power_of_2(periods_since_cliff_chk)
        prev_power_2 = prev_power_of_2(periods_since_cliff_chk)

        # we will delete a checkoint if there is an older checkpoint closer to the next power of two
        # and there is a younger checkpoint closer to the previous power of two
        if periods_since_clif_prev_chk <= next_power_2 and periods_since_cliff_next_chk >= prev_power_2:
            to_delete.append(chk)
    return to_delete
