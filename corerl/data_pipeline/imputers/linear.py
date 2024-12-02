import numpy as np

from dataclasses import dataclass
from numba import njit
from typing import Any
from omegaconf import MISSING

from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group


@dataclass
class LinearImputerConfig(BaseImputerConfig):
    name: str = "linear"
    max_gap: int = MISSING # Maximum number of NaNs between the two values used in interpolation


@dataclass
class LinearImputerTemporalState:
    prev_val: float | None = None # Last non-NaN value in the previous PipelineFrame, otherwise None
    num_nans: int = 0 # Number of NaNs after 'prev_val' in the previous PipelineFrame


class LinearImputer(BaseImputer):
    def __init__(self, cfg: LinearImputerConfig, tag_cfg: Any):
        super().__init__(cfg, tag_cfg)
        self.max_gap = cfg.max_gap

    def __call__(self, pf: PipelineFrame, tag: str):
        ts = pf.temporal_state.get(StageCode.IMPUTER)
        ts = ts or {}
        assert isinstance(ts, dict)

        tag_ts = ts.get(tag, LinearImputerTemporalState())
        ts[tag] = tag_ts
        assert isinstance(tag_ts, LinearImputerTemporalState)

        tag_data = pf.data[tag].to_numpy()

        prev = np.nan if tag_ts.prev_val is None else tag_ts.prev_val
        new_tag_data, new_prev, new_num_nans = linear_interpolation(tag_data, prev, tag_ts.num_nans, self.max_gap)

        tag_ts.prev_val = new_prev
        tag_ts.num_nans = new_num_nans

        pf.data[tag] = new_tag_data
        pf.temporal_state[StageCode.IMPUTER] = ts

        return pf


imputer_group.dispatcher(LinearImputer)



@njit
def impute_gap(backtrack_val: float,
               lookahead_val: float,
               num_nans: int) -> np.ndarray:
    """
    Use linear interpolation to impute all of the values between 'backtrack_val
    and 'lookahead_val'. Imputed values returned as a numpy array.
    """
    bias = np.ones(num_nans) * backtrack_val
    slope = float(lookahead_val - backtrack_val) / float(num_nans + 1)
    delta = np.arange(1, num_nans + 1) * slope
    imputed_vals = bias + delta

    return imputed_vals


@njit
def linear_interpolation(x: np.ndarray, prev: float, num_nans: int, max_gap: int) -> tuple[np.ndarray, float, int]:
    """
    Traverse 'x' and replace NaNs with values imputed using linear interpolation.
    A given index's value can only be imputed if the first non-NaN value before and after
    the index are separated by at most 'max_gap' NaNs
    """
    insert_ind = 0
    for i in range(len(x)):
        v = x[i]
        assert isinstance(v, float)
        reset_prev = False
        if np.isnan(prev):
            # Can't perform linear interpolation while 'prev' is NaN
            reset_prev = True
        elif np.isnan(v):
            # If 'prev' isn't NaN, increment 'num_nans' when you encounter a NaN
            num_nans += 1
            if num_nans > max_gap:
                # When 'num_nans' exceeds 'max_gap', can't perform interpolation with current 'prev'
                reset_prev = True
        elif num_nans > 0:
            # 'v' and 'prev' are within 'max_gap' so perform interpolation
            imputed_vals = impute_gap(prev, v, num_nans)
            # impute_gap() can impute values for indices in the temporal state. Drop those imputed values
            x[insert_ind: i] = imputed_vals[-(i - insert_ind):]
            reset_prev = True
        else:
            # If 'v' and 'prev' are adjacent, don't bother interpolating
            reset_prev = True

        if reset_prev:
            prev = v
            insert_ind = i + 1
            num_nans = 0

    return x, prev, num_nans
