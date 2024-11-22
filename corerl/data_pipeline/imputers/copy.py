from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import PipelineFrame, MissingType
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group


@dataclass
class CopyImputerConfig(BaseImputerConfig):
    name: str = "copy"
    default_val: float = 0.0


class CopyImputer(BaseImputer):
    def __init__(self, cfg: CopyImputerConfig):
        super().__init__(cfg)
        self.default_val = cfg.default_val

    def _get_copy(self, data: pd.Series | pd.DataFrame, ind: pd.Timestamp) -> float:
        first_valid_ind = data.first_valid_index()
        assert isinstance(first_valid_ind, pd.Timestamp | None)
        if first_valid_ind is None:
            return self.default_val
        elif ind < first_valid_ind:
            # If there are NaNs at the beginning of the Series,
            # replace with the first non-NaN entry
            copy_index = data.first_valid_index()
        else:
            # Otherwise, replace with the previous non-NaN entry
            copy_index = data.loc[first_valid_ind : ind].last_valid_index()

        return float(data.loc[copy_index])


    def _get_copies(self, data: pd.Series | pd.DataFrame, inds: pd.DatetimeIndex) -> np.ndarray:
        copied_vals = []
        for ind in inds:
            copied_val = self._get_copy(data, ind)
            copied_vals.append(copied_val)

        return np.array(copied_vals)

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        data = pf.data
        missing_info = pf.missing_info
        tag_data = data[tag]
        tag_missing_info = missing_info[tag]

        missing_inds: Any = tag_missing_info.index[tag_missing_info > MissingType.NULL]
        if len(missing_inds) == 0:
            return pf
        else:
            assert isinstance(missing_inds, pd.DatetimeIndex)
            copied_vals = self._get_copies(tag_data, missing_inds)
            data.loc[missing_inds, tag] = copied_vals
            return pf


imputer_group.dispatcher(CopyImputer)
