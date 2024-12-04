from dataclasses import dataclass
from typing import Dict

import numpy as np

from corerl.data.online_stats.exp_moving import ExpMovingAvg, ExpMovingVar
from corerl.data_pipeline.datatypes import MissingType, PipelineFrame, StageCode
from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group
from corerl.data_pipeline.utils import update_missing_info_col


@dataclass
class EMAFilterConfig(BaseOddityFilterConfig):
    name: str = "exp_moving"
    alpha: float = 0.99
    tolerance: float = 2.0
    warmup: int = 10  #  number of warmup steps before rejecting


@dataclass
class EMAFilterTemporalState:
    ema: ExpMovingAvg
    emv: ExpMovingVar
    n_obs: int = 0


type TagName = str


class EMAFilter(BaseOddityFilter):
    """
    Uses exponential moving average and variance to detect outliers
    Expected usage is to feed streams of dataframes to this class's filter function.
    Dataframes can have any number of columns, or rows.

    This class will keep exponential moving stats for each column.
    If any value in a given column has distance from its moving avg greater than 2*sigma,
    (where sigma is the moving std) then it will be considered an outlier and
    it will be replaced by a NaN.
    """

    def __init__(self, cfg: EMAFilterConfig) -> None:
        super().__init__(cfg)
        self.alpha = cfg.alpha
        self.tolerance = cfg.tolerance
        self.warmup = cfg.warmup

    def _get_tag_ts(self, stage_ts: Dict[TagName, EMAFilterTemporalState], tag: TagName) -> EMAFilterTemporalState:
        """
        Gets tag temporal state. If tag temporal state is not initialized, sets default
        and updates the stage temporal state.
        This is a class method because it uses self.alpha in construction of the default temporal state.
        """
        tag_ts = stage_ts.get(tag, EMAFilterTemporalState(ExpMovingAvg(self.alpha), ExpMovingVar(self.alpha)))
        stage_ts[tag] = tag_ts
        return tag_ts

    def __call__(self, pf: PipelineFrame, tag: str, update_stats: bool = True) -> PipelineFrame:
        """
        If update_stats is True, data in the DataFrame is used to update
        the running statistics. It may not be desirable to update the running
        statistics if, for example, historical data should be re-processed with
        the most up-to-date running statistics.
        """
        stage_ts = _get_ts(pf)
        tag_ts = self._get_tag_ts(stage_ts, tag)
        tag_data = pf.data[tag].to_numpy()

        if update_stats:
            _update_stats(tag_data, tag_ts)

        oddity_mask = _get_oddity_mask(x=tag_data, mu=tag_ts.ema(), var=tag_ts.emv(), tolerance=self.tolerance)
        post_warmup_mask = _get_post_warmup_mask(prev_n_obs=tag_ts.n_obs, data_len=len(tag_data), warmup=self.warmup)
        filter_mask = oddity_mask & post_warmup_mask

        update_missing_info_col(
            missing_info=pf.missing_info, name=tag, missing_type_mask=filter_mask, new_val=MissingType.OUTLIER
        )
        tag_data[filter_mask] = np.nan
        tag_ts.n_obs += len(tag_data)

        return pf


outlier_group.dispatcher(EMAFilter)


def _get_post_warmup_mask(prev_n_obs: int, data_len: int, warmup: int) -> np.ndarray:
    """
    Returns mask with False for points before warmup period and True after
    """
    n_observed = np.array([prev_n_obs + i for i in range(data_len)])
    post_warmup_mask = n_observed >= warmup

    return post_warmup_mask


def _get_ts(pf: PipelineFrame) -> Dict[TagName, EMAFilterTemporalState]:
    ts = pf.temporal_state.get(StageCode.ODDITY)
    ts = ts or {}
    assert isinstance(ts, dict)
    pf.temporal_state[StageCode.ODDITY] = ts
    return ts


def _update_stats(tag_data: np.ndarray, tag_ts: EMAFilterTemporalState):
    tag_ts.ema.feed(tag_data)
    tag_ts.emv.feed(tag_data)


def _get_oddity_mask(x: np.ndarray, mu: float, var: float, tolerance: float) -> np.ndarray:
    """
    Columns of df are mutable, this function takes a Series
    and mutates the data (by possible setting some values to NaN)
    """
    std = np.sqrt(var)
    oddity_mask = np.abs(mu - x) > tolerance * std

    return oddity_mask
