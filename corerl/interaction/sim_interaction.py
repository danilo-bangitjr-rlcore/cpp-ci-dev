import logging
from typing import Literal

import numpy as np

from corerl.agent.base import BaseAgent
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.interaction.interaction import Interaction
from corerl.state import AppState

logger = logging.getLogger(__file__)


@config()
class SimInteractionConfig:
    name: Literal["sim_interaction"] = "sim_interaction"


class SimInteraction(Interaction):
    def __init__(
        self,
        cfg: SimInteractionConfig,
        app_state: AppState,
        agent: BaseAgent,
        env: AsyncEnv,
        pipeline: Pipeline,
    ):
        self._pipeline = pipeline
        self._env = env
        self._agent = agent
        self._app_state = app_state

        self._column_desc = pipeline.column_descriptions

        self._should_reset = True
        self._last_state: np.ndarray | None = None
        self._last_reward: float | None = None
        self._pipeline.register_hook(CallerCode.ONLINE, StageCode.SC, self._capture_last_state)
        self._pipeline.register_hook(CallerCode.ONLINE, StageCode.RC, self._capture_last_reward)



    def step(self):
        o = self._env.get_latest_obs()
        pr = self._pipeline(o, caller_code=CallerCode.ONLINE, reset_temporal_state=self._should_reset)
        assert pr.transitions is not None
        r = self._get_latest_reward()
        assert r is not None
        self._app_state.metrics.write(
            metric='reward',
            value=r,
        )
        self._agent.update_buffer(pr.transitions)
        self._agent.update()

        self._should_reset = bool(o['truncated'].any() or o['terminated'].any())

        s = self._get_latest_state()
        assert s is not None
        a = self._agent.get_action(s)
        self._env.emit_action(a)



    # ---------
    # internals
    # ---------

    def _capture_last_state(self, pf: PipelineFrame):
        if pf.caller_code != CallerCode.ONLINE:
            return

        row = pf.data.tail(1)

        self._last_state = (
            row[self._column_desc.state_cols]
            .iloc[0]
            .to_numpy(dtype=np.float32)
        )

    def _capture_last_reward(self, pf: PipelineFrame):
        if pf.caller_code != CallerCode.ONLINE:
            return

        row = pf.data.tail(1)

        self._last_reward = float(
            row["reward"]
            .iloc[0]
        )


    def _get_latest_state(self) -> np.ndarray | None:
        if self._last_state is None:
            logger.error("Tried to get interaction state, but none existed")
            return None

        return self._last_state


    def _get_latest_reward(self) -> float | None:
        if self._last_reward is None:
            logger.error("Tried to get interaction reward, but none existed")

        return self._last_reward
