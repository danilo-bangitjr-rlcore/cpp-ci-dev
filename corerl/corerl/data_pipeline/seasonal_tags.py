import pandas as pd

from corerl.data_pipeline.datatypes import MissingType, PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig, TagType


class SeasonalTagIncluder:
    def __init__(self, tag_cfgs: list[TagConfig]):
        self.has_day_of_year = any(tag_cfg.type == TagType.day_of_year for tag_cfg in tag_cfgs)
        self.has_day_of_week = any(tag_cfg.type == TagType.day_of_week for tag_cfg in tag_cfgs)
        self.has_time_of_day = any(tag_cfg.type == TagType.time_of_day for tag_cfg in tag_cfgs)

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        timestamps = pf.data.index
        assert isinstance(timestamps, pd.DatetimeIndex)

        if self.has_day_of_year:
            pf.data["day_of_year"] = timestamps.dayofyear
            pf.missing_info["day_of_year"] = [MissingType.NULL] * len(pf.data)

        if self.has_day_of_week:
            pf.data["day_of_week"] = timestamps.weekday
            pf.missing_info["day_of_week"] = [MissingType.NULL] * len(pf.data)

        if self.has_time_of_day:
            day_seconds = (timestamps.hour * 3600) + (timestamps.minute * 60) + timestamps.second
            pf.data["time_of_day"] = day_seconds
            pf.missing_info["time_of_day"] = [MissingType.NULL] * len(pf.data)

        return pf
