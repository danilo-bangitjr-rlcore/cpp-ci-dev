import numpy as np

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.tags.components.bounds import FloatBounds


def _get_oob_mask(data: np.ndarray, tag: str, bounds: FloatBounds, prep: Preprocessor, tol: float) -> np.ndarray:
    lo, hi = bounds
    b = (
        lo if lo is not None else -np.inf,
        hi if hi is not None else np.inf,
    )
    b = prep.normalize(tag, np.array(b))
    return (data < b[0] - tol) | (data > b[1] + tol)

def bound_checker(pf: PipelineFrame, tag: str, bounds: FloatBounds, prep: Preprocessor, tol: float) -> PipelineFrame:
    data = pf.data

    tag_data = data[tag].to_numpy()
    if tag_data.dtype == np.bool_:
        return pf

    # Get OOB mask
    oob_mask = _get_oob_mask(tag_data, tag, bounds, prep, tol)

    # Set OOB to NaN
    data.loc[oob_mask, tag] = np.nan

    return pf

def bound_checker_builder(bounds: FloatBounds, prep: Preprocessor, tol: float):
    def _inner(pf: PipelineFrame, tag: str):
        return bound_checker(pf, tag, bounds, prep, tol)

    return _inner
