import numpy as np

from dataclasses import dataclass
from numba import njit
from omegaconf import MISSING
from typing import Any

from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group


@dataclass
class CopyImputerConfig(BaseImputerConfig):
    name: str = "copy"
    imputation_horizon: int = MISSING


@dataclass
class CopyImputerTemporalState:
    prev_val: float | None = None
    prev_horizon: int = 0


class CopyImputer(BaseImputer):
    def __init__(self, cfg: CopyImputerConfig, tag_cfg: Any):
        super().__init__(cfg, tag_cfg)
        self.imputation_horizon = cfg.imputation_horizon

    def __call__(self, pf: PipelineFrame, tag: str):
        ts = pf.temporal_state.get(StageCode.IMPUTER)
        ts = ts or {}
        assert isinstance(ts, dict)

        tag_ts = ts.get(tag, CopyImputerTemporalState())
        ts[tag] = tag_ts
        assert isinstance(tag_ts, CopyImputerTemporalState)

        tag_data = pf.data[tag].to_numpy()

        prev = np.nan if tag_ts.prev_val is None else tag_ts.prev_val
        forward, new_prev, new_hor = copy_forward(tag_data, prev, self.imputation_horizon, tag_ts.prev_horizon)
        backward, _, _ = copy_forward(forward[::-1], np.nan, self.imputation_horizon, 0)

        tag_ts.prev_val = new_prev
        tag_ts.prev_horizon = new_hor

        pf.data[tag] = backward[::-1]
        pf.temporal_state[StageCode.IMPUTER] = ts

        return pf


imputer_group.dispatcher(CopyImputer)


@njit
def copy_forward(x: np.ndarray, prev: float, horizon: int, prev_horizon: int):
    out = np.zeros_like(x)

    num_nans = prev_horizon
    for i in range(len(x)):
        v = x[i]
        if np.isnan(v):
            num_nans += 1

            if num_nans <= horizon:
                out[i] = prev
            else:
                out[i] = np.nan

        else:
            num_nans = 0
            out[i] = v
            prev = v

    return out, prev, num_nans
