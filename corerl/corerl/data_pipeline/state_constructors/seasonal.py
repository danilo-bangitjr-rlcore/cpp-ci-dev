from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from lib_config.config import MISSING, computed, config

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagType

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class SeasonalConfig:
    time_of_day_enabled: bool = MISSING
    day_of_week_enabled: bool = MISSING
    time_of_year_enabled: bool = MISSING

    @computed('time_of_day_enabled')
    @classmethod
    def _time_of_day_enabled(cls, cfg: 'MainConfig'):
        return any(tag_cfg.type == TagType.time_of_day for tag_cfg in cfg.pipeline.tags)

    @computed('day_of_week_enabled')
    @classmethod
    def _day_of_week_enabled(cls, cfg: 'MainConfig'):
        return any(tag_cfg.type == TagType.day_of_week for tag_cfg in cfg.pipeline.tags)

    @computed('time_of_year_enabled')
    @classmethod
    def _time_of_year_enabled(cls, cfg: 'MainConfig'):
        return any(tag_cfg.type == TagType.day_of_year for tag_cfg in cfg.pipeline.tags)

def add_seasonal_features(cfg: SeasonalConfig, pf: PipelineFrame) -> PipelineFrame:
    if not (cfg.time_of_day_enabled or cfg.day_of_week_enabled or cfg.time_of_year_enabled):
        return pf

    timestamps = pf.data.index
    assert isinstance(timestamps, pd.DatetimeIndex)
    num_entries = len(timestamps)

    if cfg.time_of_day_enabled:
        pf.data["time_of_day_sin"] = np.sin(2.0 * np.pi * pf.data["time_of_day"] / 86400.0)
        pf.data["time_of_day_cos"] = np.cos(2.0 * np.pi * pf.data["time_of_day"] / 86400.0)

    if cfg.day_of_week_enabled:
        weekday_cols = np.zeros((num_entries, 7))
        weekday_cols[range(num_entries), pf.data["day_of_week"].astype(int)] = 1.0
        col_names = [f"day_of_week_{i}" for i in range(7)]
        pf.data[col_names] = weekday_cols

    if cfg.time_of_year_enabled:
        days_in_year = 366.0 * np.ones(num_entries)
        not_leap_year = ~ timestamps.is_leap_year
        days_in_year -= not_leap_year
        pf.data["time_of_year_sin"] = np.sin(2.0 * np.pi * pf.data["day_of_year"] / days_in_year)
        pf.data["time_of_year_cos"] = np.cos(2.0 * np.pi * pf.data["day_of_year"] / days_in_year)

    return pf
