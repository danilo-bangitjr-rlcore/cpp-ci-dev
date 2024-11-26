import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any
from omegaconf import MISSING

from corerl.data_pipeline.datatypes import PipelineFrame, MissingType
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group
from corerl.data_pipeline.tag_config import TagConfig


@dataclass
class LinearImputerConfig(BaseImputerConfig):
    name: str = "linear"
    imputation_horizon: int = MISSING


class LinearImputer(BaseImputer):
    def __init__(self, cfg: LinearImputerConfig, **kwargs):
        super().__init__(cfg)
        self.imputation_horizon = cfg.imputation_horizon
        assert 'tag_cfg' in kwargs
        assert isinstance(kwargs['tag_cfg'], TagConfig)
        self.bounds = kwargs['tag_cfg'].bounds

    def _get_slope(self,
                   first_ind_int: int,
                   second_ind_int: int,
                   first_val: float,
                   second_val: float) -> float:
        """
        first_ind_int and second_ind_int (and their corresponding values) should be in chronological order
        """
        return float((second_val - first_val) / (second_ind_int - first_ind_int))

    def _lookahead_imputation(self,
                              data: pd.Series | pd.DataFrame,
                              impute_ind_int: int,
                              first_lookahead_ind: pd.Timestamp,
                              max_lookahead: int) -> float:
        """
        If there are only non-NaN value(s) ahead of the impute_ind that are within the imputation_horizon,
        perform linear interpolation if there are at least two values in the lookahead
        or perform copy imputation if there is only one non-NaN value in the lookahead
        """
        first_lookahead_ind_int = data.index.get_loc(first_lookahead_ind)
        assert isinstance(first_lookahead_ind_int, int)
        first_lookahead_val = data.loc[first_lookahead_ind]
        second_lookahead_ind = data.iloc[first_lookahead_ind_int + 1: max_lookahead].first_valid_index()
        if second_lookahead_ind is None:
            # Only a single non-NaN value in lookahead so use copy-imputation
            imputed_val = first_lookahead_val
        else:
            # Use two non-NaN values in lookahead to do linear interpolation
            second_lookahead_ind_int = data.index.get_loc(second_lookahead_ind)
            assert isinstance(second_lookahead_ind_int, int)
            second_lookahead_val = data.loc[second_lookahead_ind]
            slope = self._get_slope(first_lookahead_ind_int,
                                    second_lookahead_ind_int,
                                    first_lookahead_val,
                                    second_lookahead_val)
            imputed_val = slope * (impute_ind_int - first_lookahead_ind_int) + first_lookahead_val
            imputed_val = np.clip(imputed_val, self.bounds[0], self.bounds[1])

        return float(imputed_val)

    def _backtrack_imputation(self,
                              data: pd.Series | pd.DataFrame,
                              impute_ind_int: int,
                              first_backtrack_ind: pd.Timestamp,
                              max_backtrack: int) -> float:
        """
        If there are only non-NaN value(s) before the impute_ind that are within the imputation_horizon,
        perform linear interpolation if there are at least two values in the backtrack
        or perform copy imputation if there is only one non-NaN value in the backtrack
        """
        first_backtrack_ind_int = data.index.get_loc(first_backtrack_ind)
        assert isinstance(first_backtrack_ind_int, int)
        first_backtrack_val = data.loc[first_backtrack_ind]
        second_backtrack_ind = data.iloc[max_backtrack: first_backtrack_ind_int].last_valid_index()
        if second_backtrack_ind is None:
            # Only a single non-NaN value in backtrack so use copy-imputation
            imputed_val = first_backtrack_val
        else:
            # Use two non-NaN values in backtrack to do linear interpolation
            second_backtrack_ind_int = data.index.get_loc(second_backtrack_ind)
            assert isinstance(second_backtrack_ind_int, int)
            second_backtrack_val = data.loc[second_backtrack_ind]
            slope = self._get_slope(second_backtrack_ind_int,
                                    first_backtrack_ind_int,
                                    second_backtrack_val,
                                    first_backtrack_val)
            imputed_val = slope * (impute_ind_int - first_backtrack_ind_int) + first_backtrack_val
            imputed_val = np.clip(imputed_val, self.bounds[0], self.bounds[1])

        return float(imputed_val)

    def _straddle_imputation(self,
                             data: pd.Series | pd.DataFrame,
                             impute_ind_int: int,
                             first_backtrack_ind: pd.Timestamp,
                             first_lookahead_ind: pd.Timestamp) -> float:
        """
        Standard linear interpolation when there is a non-NaN value before and after the impute_ind
        that are within the imputation horizon
        """
        first_backtrack_ind_int = data.index.get_loc(first_backtrack_ind)
        assert isinstance(first_backtrack_ind_int, int)
        first_backtrack_val = data.loc[first_backtrack_ind]
        first_lookahead_ind_int = data.index.get_loc(first_lookahead_ind)
        assert isinstance(first_lookahead_ind_int, int)
        first_lookahead_val = data.loc[first_lookahead_ind]
        slope = self._get_slope(first_backtrack_ind_int,
                                first_lookahead_ind_int,
                                first_backtrack_val,
                                first_lookahead_val)
        imputed_val = slope * (impute_ind_int - first_backtrack_ind_int) + first_backtrack_val
        imputed_val = np.clip(imputed_val, self.bounds[0], self.bounds[1])

        return float(imputed_val)

    def _get_imputed_val(self,
                         data: pd.Series | pd.DataFrame,
                         impute_ind: pd.Timestamp) -> float:
        impute_ind_int = data.index.get_loc(impute_ind)
        assert isinstance(impute_ind_int, int)
        first_valid_ind = data.first_valid_index()
        assert isinstance(first_valid_ind, pd.Timestamp | None)
        if first_valid_ind is None:
            # The entire pd.Series consists of NaNs
            # Can't perform imputation so keep as NaNs and trigger data gap
            imputed_val = np.nan
        else:
            max_backtrack = np.clip(impute_ind_int - self.imputation_horizon, 0, len(data))
            first_backtrack_ind = data.iloc[max_backtrack : impute_ind_int].last_valid_index()
            max_lookahead = np.clip(impute_ind_int + self.imputation_horizon + 1, 0, len(data))
            first_lookahead_ind = data.iloc[impute_ind_int : max_lookahead].first_valid_index()
            if first_backtrack_ind is None and first_lookahead_ind is None:
                # No non-NaN values within imputation horizon
                imputed_val = np.nan
            elif first_backtrack_ind is None:
                # Only non-NaN value(s) in the lookahead
                imputed_val = self._lookahead_imputation(data, impute_ind_int, first_lookahead_ind, max_lookahead)
            elif first_lookahead_ind is None:
                # Only non-NaN value(s) in the backtrack
                imputed_val = self._backtrack_imputation(data, impute_ind_int, first_backtrack_ind, max_backtrack)
            else:
                # Can perform linear interpolation using a value before and a value after the missing ind
                imputed_val = self._straddle_imputation(data, impute_ind_int, first_backtrack_ind, first_lookahead_ind)

        return imputed_val

    def _get_imputed_vals(self, data: pd.Series | pd.DataFrame, imputed_inds: pd.DatetimeIndex) -> np.ndarray:
        imputed_vals = []
        for ind in imputed_inds:
            imputed_val = self._get_imputed_val(data, ind)
            imputed_vals.append(imputed_val)

        return np.array(imputed_vals)

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
            imputed_vals = self._get_imputed_vals(tag_data, missing_inds)
            data.loc[missing_inds, tag] = imputed_vals
            return pf


imputer_group.dispatcher(LinearImputer)
