import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import MissingType, PipelineFrame
from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group
from corerl.data_pipeline.utils import update_missing_info
from corerl.state import AppState

logger = logging.getLogger(__name__)


@config()
class StuckDetectorConfig(BaseOddityFilterConfig):
    name: Literal["stuck_detector"] = "stuck_detector"
    eps: float = 1e-3
    step_tol: int = 10  #  number of steps before marking stuck


@dataclass
class StuckDetectorTemporalState:
    prev_val: float = np.nan
    stuck_steps: int = 0


class StuckDetector(BaseOddityFilter):
    """
    Checks if a sensor is "stuck" by checking if the value has never deviated
    by at least epsilon from its previous value
    """

    def __init__(self, cfg: StuckDetectorConfig, app_state: AppState) -> None:
        super().__init__(cfg, app_state)
        self.eps = cfg.eps
        self.step_tol = cfg.step_tol

    def __call__(self, pf: PipelineFrame, tag: str, ts: object | None, update_stats: bool = True):

        tag_ts = ts if ts is not None else StuckDetectorTemporalState()
        assert isinstance(tag_ts, StuckDetectorTemporalState)
        tag_data = pf.data[tag].to_numpy()

        oddity_mask = np.full(tag_data.shape, False)
        i = 0
        while i < len(tag_data):
            x_i = tag_data[i]
            if not np.isnan(x_i):
                val_changed = np.isnan(tag_ts.prev_val) or abs(tag_ts.prev_val - x_i) > self.eps
                tag_ts.prev_val = x_i
                if val_changed:
                    tag_ts.stuck_steps = 0
                else:
                    tag_ts.stuck_steps += 1

            if tag_ts.stuck_steps >= self.step_tol:
                oddity_mask[i] = True
            i += 1

        # update missing info and set nans
        update_missing_info(pf.missing_info, name=tag, missing_mask=oddity_mask, new_val=MissingType.OUTLIER)
        tag_data[oddity_mask] = np.nan

        return pf, tag_ts


outlier_group.dispatcher(StuckDetector)
