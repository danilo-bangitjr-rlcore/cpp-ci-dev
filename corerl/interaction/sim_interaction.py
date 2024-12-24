import logging
import numpy as np

from corerl.agent.base import BaseAgent
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv


logger = logging.getLogger(__file__)

class SimInteraction:
    def __init__(
        self,
        agent: BaseAgent,
        env: AsyncEnv,
        pipeline: Pipeline,
        tag_configs: list[TagConfig],
    ):
        self._pipeline = pipeline
        self._env = env
        self._agent = agent

        self._non_state_tags = set(
            tag.name
            for tag in tag_configs
            if tag.tag_type != "observation"
        )

        self._should_reset = True
        self._last_state: np.ndarray | None = None
        self._pipeline.register_hook(StageCode.SC, self._capture_last_state)


    def step(self):
        o = self._env.get_latest_obs()
        pr = self._pipeline(o, caller_code=CallerCode.ONLINE, reset_temporal_state=self._should_reset)
        assert pr.transitions is not None
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

        tags = set(row.columns) - self._non_state_tags
        state = row[list(tags)].iloc[0].to_numpy()

        self._last_state = np.asarray(state, dtype=np.float32)


    def _get_latest_state(self) -> np.ndarray | None:
        if self._last_state is None:
            logger.error("Tried to get interaction state, but none existed")
            return None

        return self._last_state
