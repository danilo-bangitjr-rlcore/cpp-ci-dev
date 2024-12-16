import numpy as np

from corerl.data_pipeline.datatypes import MissingType, PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.utils import update_missing_info


def _get_oob_mask(data: np.ndarray, cfg: TagConfig) -> np.ndarray:
    if cfg.bounds[0] is None:
        lower_bound_mask = np.array([False] * len(data))
    else:
        lower_bound_mask = data < cfg.bounds[0]

    if cfg.bounds[1] is None:
        upper_bound_mask = np.array([False] * len(data))
    else:
        upper_bound_mask = data > cfg.bounds[1]

    oob_mask = lower_bound_mask | upper_bound_mask

    return oob_mask

def bound_checker(pf: PipelineFrame, tag: str, cfg: TagConfig) -> PipelineFrame:
    data = pf.data

    if data.shape[0] == 0:
        # empty dataframe, do nothing
        return pf

    tag_data = data[tag].to_numpy()
    if tag_data.dtype == np.bool_:
        return pf

    # Get OOB mask
    oob_mask = _get_oob_mask(tag_data, cfg)

    # Set OOB to NaN
    data.loc[oob_mask, tag] = np.nan

    # Update pf.missing_info
    update_missing_info(pf.missing_info, tag, oob_mask, MissingType.BOUNDS)

    return pf


def bound_checker_builder(cfg: TagConfig):
    def _inner(pf: PipelineFrame, tag: str):
        return bound_checker(pf, tag, cfg)

    return _inner
