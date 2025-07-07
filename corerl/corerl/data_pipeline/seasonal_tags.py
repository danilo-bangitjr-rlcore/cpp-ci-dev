from collections.abc import Sequence

import pandas as pd
from lib_defs.config_defs.tag_config import TagType
from lib_utils.maybe import Maybe

from corerl.data_pipeline.datatypes import MissingType, PipelineFrame
from corerl.tags.seasonal import SeasonalTags
from corerl.tags.tag_config import TagConfig


class SeasonalTagIncluder:
    def __init__(self, tag_cfgs: Sequence[TagConfig]):
        seasonal_tags = [tag_cfg for tag_cfg in tag_cfgs if tag_cfg.type == TagType.seasonal]

        self.has_day_of_year = (
            Maybe.find(lambda tag_cfg: tag_cfg.name == SeasonalTags.day_of_year, seasonal_tags)
            .is_some()
        )
        self.has_day_of_week = (
            Maybe.find(lambda tag_cfg: tag_cfg.name == SeasonalTags.day_of_week, seasonal_tags)
            .is_some()
        )
        self.has_time_of_day = (
            Maybe.find(lambda tag_cfg: tag_cfg.name == SeasonalTags.time_of_day, seasonal_tags)
            .is_some()
        )

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        timestamps = pf.data.index
        assert isinstance(timestamps, pd.DatetimeIndex)

        if self.has_day_of_year:
            pf.data[SeasonalTags.day_of_year] = timestamps.dayofyear
            pf.missing_info[SeasonalTags.day_of_year] = [MissingType.NULL] * len(pf.data)

        if self.has_day_of_week:
            pf.data[SeasonalTags.day_of_week] = timestamps.weekday
            pf.missing_info[SeasonalTags.day_of_week] = [MissingType.NULL] * len(pf.data)

        if self.has_time_of_day:
            day_seconds = (timestamps.hour * 3600) + (timestamps.minute * 60) + timestamps.second
            pf.data[SeasonalTags.time_of_day] = day_seconds
            pf.missing_info[SeasonalTags.time_of_day] = [MissingType.NULL] * len(pf.data)

        return pf
