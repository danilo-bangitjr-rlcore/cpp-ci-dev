from collections.abc import Sequence

import numpy as np
import pandas as pd
from lib_defs.config_defs.tag_config import TagType

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.tags.seasonal import SeasonalTags
from corerl.tags.tag_config import TagConfig


class SeasonalTagFeatures:
    def __init__(self, tag_cfgs: Sequence[TagConfig]):
        seasonal_tags = [tag_cfg for tag_cfg in tag_cfgs if tag_cfg.type == TagType.seasonal]

        self.has_day_of_year = any(tag_cfg.name == SeasonalTags.day_of_year for tag_cfg in seasonal_tags)
        self.has_day_of_week = any(tag_cfg.name == SeasonalTags.day_of_week for tag_cfg in seasonal_tags)
        self.has_time_of_day = any(tag_cfg.name == SeasonalTags.time_of_day for tag_cfg in seasonal_tags)
        self.has_second_in_hour = any(tag_cfg.name == SeasonalTags.second_in_hour for tag_cfg in seasonal_tags)

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if not (self.has_time_of_day or self.has_day_of_week or self.has_day_of_year or self.has_second_in_hour):
            return pf

        timestamps = pf.data.index
        assert isinstance(timestamps, pd.DatetimeIndex)
        num_entries = len(timestamps)

        if self.has_time_of_day:
            pf.data["time_of_day_sin"] = (np.sin(2.0 * np.pi * pf.data[SeasonalTags.time_of_day] / 86400.0) + 1) / 2
            pf.data["time_of_day_cos"] = (np.cos(2.0 * np.pi * pf.data[SeasonalTags.time_of_day] / 86400.0) + 1) / 2

        if self.has_day_of_week:
            weekday_cols = np.zeros((num_entries, 7))
            weekday_cols[range(num_entries), pf.data[SeasonalTags.day_of_week].astype(int)] = 1.0
            col_names = [f"day_of_week_{i}" for i in range(7)]
            pf.data[col_names] = weekday_cols

        if self.has_day_of_year:
            days_in_year = 366.0 * np.ones(num_entries)
            not_leap_year = ~ timestamps.is_leap_year
            days_in_year -= not_leap_year
            sin_data = (np.sin(2.0 * np.pi * pf.data[SeasonalTags.day_of_year] / days_in_year) + 1) / 2
            cos_data = (np.cos(2.0 * np.pi * pf.data[SeasonalTags.day_of_year] / days_in_year) + 1) / 2
            pf.data["time_of_year_sin"] = sin_data
            pf.data["time_of_year_cos"] = cos_data

        if self.has_second_in_hour:
            pf.data["second_in_hour_sin"] = (
                np.sin(2.0 * np.pi * pf.data[SeasonalTags.second_in_hour] / 3600.0) + 1
            ) / 2
            pf.data["second_in_hour_cos"] = (
                np.cos(2.0 * np.pi * pf.data[SeasonalTags.second_in_hour] / 3600.0) + 1
            ) / 2

        return pf
