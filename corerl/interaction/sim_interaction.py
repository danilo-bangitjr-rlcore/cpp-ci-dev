import logging
from datetime import UTC, datetime, timedelta

import pandas as pd

from corerl.agent.greedy_ac import GreedyAC
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv
from corerl.interaction.configs import InteractionConfig
from corerl.interaction.deployment_interaction import DeploymentInteraction
from corerl.messages.heartbeat import Heartbeat, HeartbeatConfig
from corerl.state import AppState
from corerl.utils.time import percent_time_elapsed

logger = logging.getLogger(__name__)

class DummyHeartbeat(Heartbeat):
    def __init__(self, cfg: HeartbeatConfig, coreio_origin: str) -> None:
        ...

    def healthcheck(self):
        ...

class SimInteraction(DeploymentInteraction):
    def __init__(
        self,
        cfg: InteractionConfig,
        app_state: AppState,
        agent: GreedyAC,
        env: DeploymentAsyncEnv,
        pipeline: Pipeline,
    ):
        super().__init__(cfg, app_state, agent, env, pipeline)
        self._last_checkpoint = datetime(1984, 1, 1, tzinfo=UTC)

    def _init_heartbeat(self):
        return DummyHeartbeat(self._cfg.heartbeat, self._env.get_cfg().coreio_origin)

    @property
    def _step_timestamp(self):
        """
        simulate passage of time as a function of agent step
        """
        return self._app_state.start_time + min(self._app_state.agent_step-1, 0) * self.obs_period


    def interact_forever(self):
        while True:
            self.step()
            yield

    # ---------
    # internals
    # ---------
    def _should_reset(self, observation: pd.DataFrame) -> bool:
        return bool(observation['truncated'].any() or observation['terminated'].any())

    def _wait_for_next_step(self):
        ...

    def _get_elapsed_guardrail_duration(self, guardrail_duration: timedelta):
        """
        rely on simulated time rather than system time
        """
        return percent_time_elapsed(
            start=self._app_state.start_time,
            end=self._app_state.start_time + guardrail_duration,
            cur=self._step_timestamp,
        )

    def _should_take_action(self, curr_time: datetime) -> bool:
        return True

    def _state_is_fresh(self):
        """
        check that a state has been previously observed
        """
        if self._last_state_timestamp is None:
            logger.error("Interaction state is None")
            return False
        return True

    # ---------------------
    # -- Historical Data --
    # ---------------------
    def warmup_pipeline(self):
        if self._cfg.warmup_period is None:
            return

        warmup_steps = int(self._cfg.warmup_period / self.obs_period)
        for _ in range(warmup_steps):
            # Get latest state
            o = self._env.get_latest_obs()
            pipe_return = self._pipeline(o, data_mode=DataMode.ONLINE, reset_temporal_state=self._should_reset(o))
            self._capture_latest_state(pipe_return)

            # Take action
            sa = self._get_latest_state_action()
            assert sa is not None
            s, action_lo, action_hi = sa
            next_a = self._agent.get_action_interaction(s, action_lo, action_hi)
            norm_next_a_df = self._pipeline.action_constructor.get_action_df(next_a)
            # clip to the normalized action bounds
            norm_next_a_df = self._clip_action_bounds(norm_next_a_df, action_lo, action_hi)
            next_a_df = self._pipeline.preprocessor.inverse(norm_next_a_df)
            self._env.emit_action(next_a_df, log_action=False)


    # -------------------
    # -- Checkpointing --
    # -------------------
    def maybe_checkpoint(self):
        now = self._last_state_timestamp
        assert now is not None
        assert self._last_checkpoint is not None
        if now - self._last_checkpoint >= self._checkpoint_freq:
            self.checkpoint()
            self._last_checkpoint = now
