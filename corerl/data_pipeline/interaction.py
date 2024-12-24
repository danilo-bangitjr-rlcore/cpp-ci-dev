import logging
import numpy as np
import pandas as pd

from datetime import UTC, datetime, timedelta
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.data_pipeline.tag_config import TagConfig


logger = logging.getLogger(__file__)

class InteractionWrapper:
    def __init__(
        self,
        pipeline: Pipeline,
        tag_configs: list[TagConfig],
        action_period: timedelta,
        tol: timedelta,
    ):
        self._pipeline = pipeline
        self._action_period = action_period
        self._tol = tol

        self._non_state_tags = set(
            tag.name
            for tag in tag_configs
            if tag.tag_type != "observation"
        )

        self._last_time: datetime | None = None
        self._last_state: np.ndarray | None = None

        self._pipeline.register_hook(StageCode.SC, self._capture_last_state)


    def _capture_last_state(self, pf: PipelineFrame):
        if pf.caller_code != CallerCode.ONLINE:
            return

        row = pf.data.tail(1)

        timestamp = row.index[0]
        assert isinstance(timestamp, pd.Timestamp)
        tags = set(row.columns) - self._non_state_tags
        state = row[list(tags)].iloc[0].to_numpy()

        self._last_time = timestamp.to_pydatetime()
        self._last_state = state


    def get_latest_state(self, time: datetime | None = None) -> np.ndarray | None:
        if time is None:
            time = datetime.now(UTC)

        if self._last_time is None:
            logger.error("Tried to get interaction state, but have no prior timestamp")
            return None

        if self._last_state is None:
            logger.error("Tried to get interaction state, but none existed")
            return None

        if np.any(np.isnan(self._last_state)):
            logger.error("Tried to get interaction state, but there were nan values")
            return None

        if time - self._last_time > self._action_period + self._tol:
            logger.error("Got a stale interaction state")
            return self._last_state

        return self._last_state
