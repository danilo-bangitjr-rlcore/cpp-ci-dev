import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, cast
from omegaconf import MISSING

from corerl.data_pipeline.datatypes import PipelineFrame, MissingType
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group, ImputerTemporalState
from corerl.data_pipeline.tag_config import TagConfig


@dataclass
class LinearImputerConfig(BaseImputerConfig):
    name: str = "linear"
    imputation_horizon: int = MISSING


class LinearImputer(BaseImputer):
    def __init__(self, cfg: LinearImputerConfig, tag_cfg: Any):
        super().__init__(cfg, tag_cfg)
        self.imputation_horizon = cfg.imputation_horizon
        assert isinstance(tag_cfg, TagConfig)
        self.bounds = tag_cfg.bounds

    def _get_ts_aware_history(self,
                              data: pd.Series | pd.DataFrame,
                              impute_ind_int: int,
                              imputer_ts: ImputerTemporalState) -> pd.Series | pd.DataFrame:
        """
        If impute_ind_int < imputation_horizon, try constructing a backward looking history of size imputation_horizon
        that comprises the indices before impute_ind in 'data' as well as the indices in the Temporal State that are
        within the imputation horizon. If ImputerTemporalState.prev_pf_data is None, simply return the previous indices
        in 'data'
        """
        if imputer_ts.prev_pf_data is None:
            return data.iloc[:impute_ind_int]

        ts_horizon = np.clip(self.imputation_horizon - impute_ind_int, 0, len(imputer_ts.prev_pf_data))
        ts_data = imputer_ts.prev_pf_data.iloc[-ts_horizon:]
        curr_pf_data = data.iloc[:impute_ind_int]
        backtrack_vals = pd.concat([ts_data, curr_pf_data])

        return backtrack_vals

    def _get_backtrack_data(self,
                            data: pd.Series | pd.DataFrame,
                            impute_ind_int: int,
                            imputer_ts: ImputerTemporalState) -> pd.Series | pd.DataFrame:
        if impute_ind_int < self.imputation_horizon:
            # Consider backtrack values in temporal state
            backtrack_vals = self._get_ts_aware_history(data, impute_ind_int, imputer_ts)
        else:
            backtrack_vals = data.iloc[impute_ind_int - self.imputation_horizon : impute_ind_int]
        assert isinstance(backtrack_vals, pd.Series | pd.DataFrame)
        backtrack_vals = cast(pd.Series | pd.DataFrame, backtrack_vals)

        return backtrack_vals

    def _get_lookahead_data(self,
                            data: pd.Series | pd.DataFrame,
                            impute_ind_int: int) -> pd.Series | pd.DataFrame:
        lookahead_horizon = np.clip(impute_ind_int + self.imputation_horizon + 1, 0, len(data))
        lookahead_vals = data.iloc[impute_ind_int + 1 : lookahead_horizon]

        return lookahead_vals

    def _get_slope(self,
                   backtrack_val: float,
                   lookahead_val: float,
                   ind_diff: int) -> float:
        return float(lookahead_val - backtrack_val) / float(ind_diff)

    def _linear_interpolation(self,
                              backtrack_vals: pd.Series | pd.DataFrame,
                              backtrack_ind: pd.Timestamp,
                              lookahead_vals: pd.Series | pd.DataFrame,
                              lookahead_ind: pd.Timestamp) -> float:
        backtrack_val = backtrack_vals.loc[backtrack_ind]
        lookahead_val = lookahead_vals.loc[lookahead_ind]
        backtrack_ind_int = backtrack_vals.index.get_loc(backtrack_ind)
        lookahead_ind_int = lookahead_vals.index.get_loc(lookahead_ind)
        assert isinstance(backtrack_ind_int, int)
        assert isinstance(lookahead_ind_int, int)
        ind_diff = (self.imputation_horizon - backtrack_ind_int) + (lookahead_ind_int + 1)
        slope = self._get_slope(backtrack_val, lookahead_val, ind_diff)
        imputed_val = slope * (self.imputation_horizon - backtrack_ind_int) + backtrack_val

        return imputed_val

    def _get_imputed_val(self,
                         data: pd.Series | pd.DataFrame,
                         impute_ind: pd.Timestamp,
                         imputer_ts: ImputerTemporalState) -> float:
        impute_ind_int = data.index.get_loc(impute_ind)
        assert isinstance(impute_ind_int, int)

        # Try finding non-NaN value before impute_ind within imputation horizon
        backtrack_vals = self._get_backtrack_data(data, impute_ind_int, imputer_ts)
        backtrack_ind = backtrack_vals.last_valid_index()
        backtrack_ind = cast(pd.Timestamp | None, backtrack_ind)

        # Try finding non-NaN value after impute_ind within imputation horizon
        lookahead_vals = self._get_lookahead_data(data, impute_ind_int)
        lookahead_ind = lookahead_vals.first_valid_index()
        lookahead_ind = cast(pd.Timestamp | None, lookahead_ind)

        if backtrack_ind is not None and lookahead_ind is not None:
            # Found non-NaN values before and after impute_ind within imputation horizon. Can perform interpolation.
            assert isinstance(backtrack_ind, pd.Timestamp)
            assert isinstance(lookahead_ind, pd.Timestamp)
            imputed_val = self._linear_interpolation(backtrack_vals, backtrack_ind, lookahead_vals, lookahead_ind)
            return imputed_val
        else:
            # Can't find non-NaN values before and after impute_ind within imputation horizon
            return np.nan

    def _get_imputed_vals(self,
                          data: pd.Series | pd.DataFrame,
                          imputed_inds: pd.DatetimeIndex,
                          imputer_ts: ImputerTemporalState) -> np.ndarray:
        imputed_vals = []
        for ind in imputed_inds:
            imputed_val = self._get_imputed_val(data, ind, imputer_ts)
            imputed_vals.append(imputed_val)

        return np.array(imputed_vals)

    def _get_new_ts_history(self,
                            data: pd.Series | pd.DataFrame,
                            imputer_ts: ImputerTemporalState) -> ImputerTemporalState:
        """
        Set the ImputerTemporalState to be the last self.imputation_horizon rows of the current PipelineFrame's
        DataFrame so that the next PipelineFrame can potentially perform Copy Imputation if the PipelineFrame's
        are temporally continuous
        """
        imputer_ts.prev_pf_data = data.iloc[-self.imputation_horizon:].copy()

        return imputer_ts

    def _inner_call(self,
                    pf: PipelineFrame,
                    tag: str,
                    imputer_ts: ImputerTemporalState | None) -> tuple[PipelineFrame, ImputerTemporalState]:
        # Make sure imputer_ts is an ImputerTemporalState
        if imputer_ts is None:
            imputer_ts = ImputerTemporalState()
        assert isinstance(imputer_ts, ImputerTemporalState)

        data = pf.data
        missing_info = pf.missing_info
        tag_data = data[tag]
        tag_missing_info = missing_info[tag]

        # Find indices with NaNs
        missing_inds: Any = tag_missing_info.index[tag_missing_info > MissingType.NULL]
        if len(missing_inds) == 0:
            # No imputation necessary
            imputer_ts = self._get_new_ts_history(tag_data, imputer_ts)
            return pf, imputer_ts

        assert isinstance(missing_inds, pd.DatetimeIndex)
        imputed_vals = self._get_imputed_vals(tag_data, missing_inds, imputer_ts)
        imputer_ts = self._get_new_ts_history(tag_data, imputer_ts)
        data.loc[missing_inds, tag] = imputed_vals

        return pf, imputer_ts


imputer_group.dispatcher(LinearImputer)
