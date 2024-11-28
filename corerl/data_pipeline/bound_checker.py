import numpy as np
import pandas as pd

from typing import Hashable
from corerl.data_pipeline.datatypes import MissingType, PipelineFrame, update_missing_info_col
from corerl.data_pipeline.tag_config import TagConfig


def _get_oob_mask(data: pd.DataFrame, name: Hashable, cfg: TagConfig) -> np.ndarray:
    np_tag_col = data[name].to_numpy()
    if cfg.bounds[0] is None:
        lower_bound_mask = np.array([False] * len(np_tag_col))
    else:
        lower_bound_mask = np_tag_col < cfg.bounds[0]

    if cfg.bounds[1] is None:
        upper_bound_mask = np.array([False] * len(np_tag_col))
    else:
        upper_bound_mask = np_tag_col > cfg.bounds[1]

    oob_mask = lower_bound_mask | upper_bound_mask

    return oob_mask

def bound_checker(pf: PipelineFrame, tag: str, cfg: TagConfig) -> PipelineFrame:
    data = pf.data

    if data.shape[0] == 0:
        # empty dataframe, do nothing
        return pf

    # Get OOB mask
    oob_mask = _get_oob_mask(data, tag, cfg)

    # Set OOB to NaN
    data.loc[oob_mask, tag] = np.nan

    # Update pf.missing_info
    update_missing_info_col(pf.missing_info, tag, oob_mask, MissingType.BOUNDS)

    return pf


def bound_checker_builder(cfg: TagConfig):
    def _inner(pf: PipelineFrame, tag: str):
        return bound_checker(pf, tag, cfg)

    return _inner
