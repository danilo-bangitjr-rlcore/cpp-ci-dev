import numpy as np
from dataclasses import dataclass
from pandas import DataFrame

from corerl.data.online_stats.exp_moving import ExpMovingAvg, ExpMovingVar
from corerl.data_pipeline.datatypes import PipelineFrame, update_missing_info_col, MissingType
from corerl.data_pipeline.outlier_detectors.base import BaseOutlierDetector, BaseOutlierDetectorConfig, outlier_group


@dataclass
class ExpMovingDetectorConfig(BaseOutlierDetectorConfig):
    name: str = "exp_moving"
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

        self.ema = ExpMovingAvg(alpha=self.alpha)
        self.emv = ExpMovingVar(alpha=self.alpha)

    def _get_outlier_mask(self, name: str, data: DataFrame, update_stats: bool) -> np.ndarray:
        """
        Columns of df are mutable, this function takes a Series
        and mutates the data (by possible setting some values to NaN)
        """
        x = data[name].to_numpy()

        # update stats
        if update_stats:
            self.ema.feed(x)
            self.emv.feed(x)

        # collect stats
        mu = self.ema()
        var = self.emv()
        std = np.sqrt(var)

        # find outliers
        outliers = np.abs(mu - x) > self.tolerance * std

        return outliers

    def _filter_col(self, name: str, pf: PipelineFrame, update_stats: bool) -> None:
        outlier_mask = self._get_outlier_mask(name=name, data=pf.data, update_stats=update_stats)

        # set outliers to NaN
        pf.data.loc[outlier_mask, name] = np.nan

        # update missing info
        update_missing_info_col(pf.missing_info, name, outlier_mask, MissingType.OUTLIER)

    def __call__(self, pf: PipelineFrame, tag: str, update_stats: bool = True) -> PipelineFrame:
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

        self._filter_col(tag, pf, update_stats)
        return pf


outlier_group.dispatcher(ExpMovingDetector)
