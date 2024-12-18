from typing import Literal
import numpy as np

from dataclasses import dataclass
from numba import njit

from corerl.configs.config import config, MISSING
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group
from corerl.data_pipeline.utils import get_tag_temporal_state


@config()
class CopyImputerConfig(BaseImputerConfig):
    name: Literal['copy'] = "copy"
    imputation_horizon: int = MISSING


@dataclass
class CopyImputerTemporalState:
    prev_val: float | None = None
    prev_horizon: int = 0


class CopyImputer(BaseImputer):
    def __init__(self, cfg: CopyImputerConfig):
        super().__init__(cfg)
        self.imputation_horizon = cfg.imputation_horizon

    def __call__(self, pf: PipelineFrame, tag: str):
        tag_ts = get_tag_temporal_state(
            StageCode.IMPUTER,
            tag,
            pf.temporal_state,
            default=CopyImputerTemporalState,
        )

        tag_data = pf.data[tag].to_numpy()

        prev = np.nan if tag_ts.prev_val is None else tag_ts.prev_val
        forward, new_prev, new_hor = copy_forward(tag_data, prev, self.imputation_horizon, tag_ts.prev_horizon)
        backward, _, _ = copy_backward(forward, np.nan, self.imputation_horizon, 0)

        tag_ts.prev_val = new_prev
        tag_ts.prev_horizon = new_hor

        pf.data[tag] = backward
        return pf


imputer_group.dispatcher(CopyImputer)


@njit
def copy_backward(x: np.ndarray, prev: float, horizon: int, prev_horizon: int):
    x = x[::-1]
    x, new_prev, new_horizon = copy_forward(x, prev, horizon, prev_horizon)
    return x[::-1], new_prev, new_horizon

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
