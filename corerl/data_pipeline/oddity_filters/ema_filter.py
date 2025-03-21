import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from corerl.configs.config import config
from corerl.data_pipeline.data_utils.exp_moving import ExpMovingAvg, ExpMovingVar
from corerl.data_pipeline.datatypes import DataMode, MissingType, PipelineFrame, StageCode
from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group
from corerl.data_pipeline.utils import get_tag_temporal_state, update_missing_info
from corerl.state import AppState

logger = logging.getLogger(__name__)


@config()
class EMAFilterConfig(BaseOddityFilterConfig):
    name: Literal["exp_moving"] = "exp_moving"
    alpha: float = 0.99
    tolerance: float = 2.0
    warmup: int = 10  #  number of warmup steps before rejecting


@dataclass
class EMAFilterTemporalState:
    ema: ExpMovingAvg
    emv: ExpMovingVar
    n_obs: int = 0
    _warmup_queue: list[float] = field(default_factory=list)


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

    def __init__(self, cfg: EMAFilterConfig, app_state: AppState) -> None:
        super().__init__(cfg, app_state)
        self.alpha = cfg.alpha
        self.tolerance = cfg.tolerance
        self.warmup = cfg.warmup

    def __call__(self, pf: PipelineFrame, tag: str, update_stats: bool = True) -> PipelineFrame:
        """
        If update_stats is True, data in the DataFrame is used to update
        the running statistics. It may not be desirable to update the running
        statistics if, for example, historical data should be re-processed with
        the most up-to-date running statistics.
        """
        tag_ts = get_tag_temporal_state(
            StageCode.ODDITY, tag, pf.temporal_state,
            default=lambda: EMAFilterTemporalState(ExpMovingAvg(self.alpha), ExpMovingVar(self.alpha)),
        )
        tag_data = pf.data[tag].to_numpy()

        oddity_mask = np.full(tag_data.shape, False)
        ema = tag_ts.ema
        emv = tag_ts.emv

        i = self._warmup(tag_data, tag_ts)
        while i < len(tag_data):
            x_i = tag_data[i]
            if not np.isnan(x_i):
                std = np.sqrt(emv.var)
                is_oddity = np.abs(ema.mu - x_i) > self.tolerance * std
                oddity_mask[i] = is_oddity
                if is_oddity:
                    logger.info(f"EMA filtered {tag} value {x_i} with mu {ema.mu} and std {std}")
            ema.feed_single(x_i)
            emv.feed_single(x_i)
            i += 1

        self._log_moving_stats(tag, tag_ts, pf.data_mode)

        # update missing info and set nans
        update_missing_info(pf.missing_info, name=tag, missing_mask=oddity_mask, new_val=MissingType.OUTLIER)
        tag_data[oddity_mask] = np.nan

        return pf

    def _log_moving_stats(self, tag: str, tag_ts: EMAFilterTemporalState, data_mode: DataMode):
        ema = tag_ts.ema
        emv = tag_ts.emv
        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric=f"pipeline_{data_mode.name}_outlier_{tag}_ema",
            value=ema.mu
        )
        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric=f"pipeline_{data_mode.name}_outlier_{tag}_emstd",
            value=np.sqrt(emv.var)
        )


    def _warmup(self, tag_data: np.ndarray, tag_ts: EMAFilterTemporalState) -> int:
        """
        Stores initial data in a queue to initialize ema and emv statistics.
        Initializes statistics upon completion of warmup.

        Returns first post-warmup index.
        """
        n_obs = tag_ts.n_obs
        warmed_up = n_obs >= self.warmup
        if warmed_up:
            return 0

        warmup_queue = tag_ts._warmup_queue

        i = 0
        while not warmed_up and i < len(tag_data):
            x_i = tag_data[i]
            if not np.isnan(x_i):
                n_obs += 1
                warmup_queue.append(x_i)
            i += 1
            warmed_up = n_obs >= self.warmup

        tag_ts.n_obs = n_obs

        if warmed_up:
            unique_vals = set(warmup_queue)
            if len(unique_vals) == 1:
                # Prevent classifying a val as an outlier due to np.mean()'s float imprecision
                # when it's the only value that's been observed thus far
                tag_ts.ema.mu = warmup_queue[0]
            else:
                tag_ts.ema.mu = np.array(warmup_queue).mean()
            tag_ts.emv.var = np.array(warmup_queue).var()

        return i


outlier_group.dispatcher(EMAFilter)
