import logging
from datetime import datetime, timedelta

import pandas as pd

from corerl.agent.greedy_ac import GreedyAC
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

    def _init_offline_chunks(self):
        ...

    def _pretrain(self):
        ...

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
        ...

    def load_historical_chunk(self):
        ...


    # -------------------
    # -- Checkpointing --
    # -------------------
    def maybe_checkpoint(self):
        ...


    def checkpoint(self):
        ...


    def restore_checkpoint(self):
        ...
