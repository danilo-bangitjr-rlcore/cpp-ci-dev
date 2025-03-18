import numpy as np

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import MissingType, PipelineFrame
from corerl.data_pipeline.utils import update_missing_info

Bounds = tuple[float | None, float | None]


def _get_oob_mask(data: np.ndarray, tag: str, bounds: Bounds, prep: Preprocessor) -> np.ndarray:
    lo, hi = bounds
    b = (
        lo if lo is not None else -np.inf,
        hi if hi is not None else np.inf,
    )
    b = prep.normalize(tag, np.array(b))
    return (data < b[0]) | (data > b[1])

def bound_checker(pf: PipelineFrame, tag: str, bounds: Bounds, prep: Preprocessor) -> PipelineFrame:
    data = pf.data

    tag_data = data[tag].to_numpy()
    if tag_data.dtype == np.bool_:
        return pf

    # Get OOB mask
    oob_mask = _get_oob_mask(tag_data, tag, bounds, prep)

    # Set OOB to NaN
    data.loc[oob_mask, tag] = np.nan

    # Update pf.missing_info
    update_missing_info(pf.missing_info, tag, oob_mask, MissingType.BOUNDS)

    return pf


def bound_checker_builder(bounds: Bounds, prep: Preprocessor):
    def _inner(pf: PipelineFrame, tag: str):
        return bound_checker(pf, tag, bounds, prep)

    return _inner
