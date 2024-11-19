from typing import Hashable, Tuple

import numpy as np
from dataclasses import dataclass
from pandas import DataFrame

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data.online_stats.exp_moving import ExpMovingAvg, ExpMovingVar
from corerl.data_pipeline.outlier_detectors.base import BaseOutlierDetector, BaseOutlierDetectorConfig, outlier_group
from corerl.data_pipeline.tag_config import TagConfig


@dataclass
class ExpMovingDetectorConfig(BaseOutlierDetectorConfig):
    name: str = 'exp_moving'
    alpha: float = 0.99
    tolerance: float = 2


class ExpMovingDetector(BaseOutlierDetector):
    """
    Uses exponential moving average and variance to detect outliers
    Expected usage is to feed streams of dataframes to this class's filter function.
    Dataframes can have any number of columns, or rows.

    This class will keep exponential moving stats for each column.
    If any value in a given column has distance from its moving avg greater than 2*sigma,
    (where sigma is the moving std) then it will be considered an outlier and
    it will be replaced by a NaN.
    """

    def __init__(self, cfg: ExpMovingDetectorConfig) -> None:
        super().__init__(cfg)
        self.alpha = cfg.alpha
        self.tolerance = cfg.tolerance

        # the following dicts store the exponential moving statistics
        # the keys are the column names in the DataFrames received by self.filter()
        self.emas: dict[Hashable, ExpMovingAvg] = {}
        self.emvs: dict[Hashable, ExpMovingVar] = {}

    def _get_stats(self, name: Hashable) -> Tuple[ExpMovingAvg, ExpMovingVar]:
        if name not in self.emas:
            self.emas[name] = ExpMovingAvg(alpha=self.alpha)
        if name not in self.emvs:
            self.emvs[name] = ExpMovingVar(alpha=self.alpha)

        return self.emas[name], self.emvs[name]

    def _filter_col(self, name: Hashable, data: DataFrame, update_stats: bool) -> None:
        """
        Columns of df are mutable, this function takes a Series
        and mutates the data (by possible setting some values to NaN)
        """
        # get stats
        ema, emv = self._get_stats(name)

        x = data[name].to_numpy()

        # update stats
        if update_stats:
            ema.feed(x)
            emv.feed(x)

        # collect stats
        mu = ema()
        var = emv()
        std = np.sqrt(var)

        # find outliers
        outliers = np.abs(mu - x) > self.tolerance * std

        # set outliers to NaN
        data.loc[outliers, name] = np.nan

    def __call__(self, pf: PipelineFrame,  cfg: TagConfig, update_stats: bool = True) -> PipelineFrame:
        """
        If update_stats is True, data in the DataFrame is used to update
        the running statistics. It may not be desirable to update the running
        statistics if, for example, historical data should be re-processed with
        the most up-to-date running statistics.
        """
        data = pf.data
        if data.shape[0] == 0:
            # empty dataframe, do nothing
            return pf

        _data = data.copy()
        for name in _data.columns:
            self._filter_col(name, _data, update_stats)

        filtered_frame = PipelineFrame(data=_data)

        return filtered_frame


outlier_group.dispatcher(ExpMovingDetector)
