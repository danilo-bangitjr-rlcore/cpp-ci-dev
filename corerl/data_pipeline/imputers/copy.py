import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any
from omegaconf import MISSING

from corerl.data_pipeline.datatypes import PipelineFrame, MissingType
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group


@dataclass
class CopyImputerConfig(BaseImputerConfig):
    name: str = "copy"
    default_val: float = 0.0
    imputation_horizon: int = MISSING


class CopyImputer(BaseImputer):
    def __init__(self, cfg: CopyImputerConfig):
        super().__init__(cfg)
        self.imputation_horizon = cfg.imputation_horizon

    def _get_imputed_val(self, data: pd.Series | pd.DataFrame, impute_ind: pd.Timestamp) -> float:
        impute_ind_int = data.index.get_loc(impute_ind)
        assert isinstance(impute_ind_int, int)
        first_valid_ind = data.first_valid_index()
        assert isinstance(first_valid_ind, pd.Timestamp | None)
        if first_valid_ind is None:
            # Only NaNs in the series
            copy_ind = None
        elif impute_ind < first_valid_ind:
            # If there are NaNs at the beginning of the Series,
            # replace with the first non-NaN entry within the imputation horizon
            max_lookahead = np.clip(impute_ind_int + self.imputation_horizon + 1, 0, len(data))
            copy_ind = data.iloc[impute_ind_int : max_lookahead].first_valid_index()
        else:
            # Otherwise, replace with the previous non-NaN entry within the imputation horizon
            max_backtrack = np.clip(impute_ind_int - self.imputation_horizon, 0, len(data))
            copy_ind = data.iloc[max_backtrack : impute_ind_int].last_valid_index()

        if copy_ind is None:
            imputed_val = np.nan
        else:
            imputed_val = data.loc[copy_ind]

        return float(imputed_val)

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        data = pf.data
        missing_info = pf.missing_info
        tag_data = data[tag]
        tag_missing_info = missing_info[tag]

        missing_inds: Any = tag_missing_info.index[tag_missing_info > MissingType.NULL]
        if len(missing_inds) == 0:
            return pf

        assert isinstance(missing_inds, pd.DatetimeIndex)
        copied_vals = self._get_imputed_vals(tag_data, missing_inds)
        data.loc[missing_inds, tag] = copied_vals
        return pf


imputer_group.dispatcher(CopyImputer)
