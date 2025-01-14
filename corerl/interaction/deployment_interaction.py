import logging
from datetime import UTC, datetime
from time import sleep

import numpy as np

from corerl.agent.base import BaseAgent
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv

logger = logging.getLogger(__file__)

class DeploymentInteraction:
    def __init__(
        self,
        agent: BaseAgent,
        env: DeploymentAsyncEnv,
        pipeline: Pipeline,
        tag_configs: list[TagConfig],
    ):
        self._pipeline = pipeline
        self._env = env
        self._agent = agent

        self._non_state_tags = set(
            tag.name
            for tag in tag_configs
            if tag.is_action or tag.is_meta
        ).union({"reward"})

        self._should_reset = True
        self._last_state: np.ndarray | None = None
        self._pipeline.register_hook(StageCode.SC, self._capture_last_state)

        self.obs_period = env.obs_period
        self.action_period = env.action_period
        self.last_obs_timestamp: datetime | None = None
        self.last_action_timestamp: datetime | None = None


    def step(self):
        self._wait_for_next_obs()
        o = self._env.get_latest_obs()
        pr = self._pipeline(o, caller_code=CallerCode.ONLINE, reset_temporal_state=self._should_reset)
        if pr.transitions is not None:
            self._agent.update_buffer(pr.transitions)
        self._agent.update()

        s = self._get_latest_state()
        assert s is not None
        if self._should_take_action():
            a = self._agent.get_action(s)
            self._env.emit_action(a)



    # ---------
    # internals
    # ---------
    def _wait_for_next_obs(self) -> None:
        now = datetime.now(UTC)
        if self.last_obs_timestamp is None:
            next_obs_timestamp = now
        else:
            next_obs_timestamp = self.last_obs_timestamp + self.obs_period

        if now >= next_obs_timestamp:
            sleep_duration = 0
        else:
            sleep_duration = (next_obs_timestamp - now).total_seconds()
        sleep(sleep_duration)
        self.last_obs_timestamp = now

    def _should_take_action(self) -> bool:
        now = datetime.now(UTC)
        take_action = False # default

        if self.last_action_timestamp is None:
            take_action = True
        elif now >= self.last_action_timestamp + self.action_period:
            take_action = True

        if take_action:
            self.last_action_timestamp = now

        return take_action


    def _capture_last_state(self, pf: PipelineFrame):
        if pf.caller_code != CallerCode.ONLINE:
            return

        row = pf.data.tail(1)

        tags = set(row.columns) - self._non_state_tags
        state = row[list(tags)].iloc[0].to_numpy()

        self._last_state = np.asarray(state, dtype=np.float32)
        logger.info(f"captured state {self._last_state}, with columns {tags}")


    def _get_latest_state(self) -> np.ndarray | None:
        if self._last_state is None:
            logger.error("Tried to get interaction state, but none existed")
            return None

        return self._last_state
