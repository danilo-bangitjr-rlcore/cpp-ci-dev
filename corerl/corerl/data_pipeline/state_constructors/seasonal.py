import numpy as np
import pandas as pd

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import PipelineFrame


@config()
class SeasonalConfig:
    time_of_day_enabled: bool = False
    day_of_week_enabled: bool = False
    time_of_year_enabled: bool = False

def add_seasonal_features(cfg: SeasonalConfig, pf: PipelineFrame) -> PipelineFrame:
    if not (cfg.time_of_day_enabled or cfg.day_of_week_enabled or cfg.time_of_year_enabled):
        return pf

    timestamps = pf.data.index
    assert isinstance(timestamps, pd.DatetimeIndex)
    num_entries = len(timestamps)

    if cfg.time_of_day_enabled:
        day_seconds = (timestamps.hour * 3600) + (timestamps.minute * 60) + timestamps.second
        pf.data["time_of_day_sin"] = np.sin(2.0 * np.pi * day_seconds / 86400.0)
        pf.data["time_of_day_cos"] = np.cos(2.0 * np.pi * day_seconds / 86400.0)

    if cfg.day_of_week_enabled:
        day_of_week = timestamps.weekday
        weekday_cols = np.zeros((num_entries, 7))
        weekday_cols[range(num_entries), day_of_week] = 1.0
        col_names = [f"day_of_week_{i}" for i in range(7)]
        pf.data[col_names] = weekday_cols

    if cfg.time_of_year_enabled:
        year_days = timestamps.dayofyear.to_numpy()
        days_in_year = 366.0 * np.ones(num_entries)
        not_leap_year = ~ timestamps.is_leap_year
        days_in_year -= not_leap_year
        pf.data["time_of_year_sin"] = np.sin(2.0 * np.pi * year_days / days_in_year)
        pf.data["time_of_year_cos"] = np.cos(2.0 * np.pi * year_days / days_in_year)

    return pf
