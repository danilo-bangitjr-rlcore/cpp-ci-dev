from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit
from pydantic.dataclasses import dataclass as config

from corerl.configs.config import MISSING
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group
from corerl.data_pipeline.utils import get_tag_temporal_state


@config(config={'extra': 'forbid'})
class LinearImputerConfig(BaseImputerConfig):
    name: Literal['linear'] = "linear"
    max_gap: int = MISSING # Maximum number of NaNs between the two values used in linear interpolation


@dataclass
class LinearImputerTemporalState:
    prev_val: float = np.nan # Last non-NaN value in the previous PipelineFrame, otherwise np.nan
    num_nans: int = 0 # Number of NaNs after 'prev_val' in the previous PipelineFrame. Can't exceed 'max_gap'


class LinearImputer(BaseImputer):
    def __init__(self, cfg: LinearImputerConfig):
        super().__init__(cfg)
        self.max_gap = cfg.max_gap

    def __call__(self, pf: PipelineFrame, tag: str):
        tag_ts = get_tag_temporal_state(
            StageCode.IMPUTER,
            tag,
            pf.temporal_state,
            default=LinearImputerTemporalState,
        )

        tag_data = pf.data[tag].to_numpy()
        new_tag_data, new_prev, new_num_nans = linear_interpolation(
            tag_data,
            tag_ts.prev_val,
            tag_ts.num_nans,
            self.max_gap,
        )

        tag_ts.prev_val = new_prev
        tag_ts.num_nans = new_num_nans

        pf.data[tag] = new_tag_data
        return pf


imputer_group.dispatcher(LinearImputer)


@njit
def impute_gap(backtrack_val: float, lookahead_val: float, num_nans: int) -> np.ndarray:
    """
    Use linear interpolation to impute all of the values between 'backtrack_val' and 'lookahead_val'
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
    insert_ind = 0 # Where to insert the imputed list of vals in 'x'
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
                # When 'num_nans' exceeds 'max_gap', can't perform linear interpolation with current 'prev'
                reset_prev = True
        elif num_nans > 0 and i - insert_ind > 0:
            # 'v' and 'prev' are within 'max_gap' so perform linear interpolation
            imputed_vals = impute_gap(prev, v, num_nans)
            # impute_gap() can impute values for NaNs that were in the temporal state (i.e if the passed 'num_nans' > 0)
            # Drop those imputed values
            x[insert_ind : i] = imputed_vals[-(i - insert_ind):]
            reset_prev = True
        else:
            # If 'v' and 'prev' are adjacent, don't bother interpolating
            reset_prev = True

        if reset_prev:
            prev = v
            insert_ind = i + 1
            num_nans = 0

    return x, prev, num_nans
