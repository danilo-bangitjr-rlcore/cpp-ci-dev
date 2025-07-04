from enum import StrEnum
from typing import Literal

from lib_config.config import MISSING, config, post_processor
from lib_defs.config_defs.tag_config import TagType

from corerl.data_pipeline.transforms import NukeConfig
from corerl.tags.base import GlobalTagAttributes


class SeasonalTags(StrEnum):
    day_of_year = "day_of_year"
    day_of_week = "day_of_week"
    time_of_day = "time_of_day"


@config()
class SeasonalTagConfig(GlobalTagAttributes):
    name: SeasonalTags = MISSING
    type: Literal[TagType.seasonal] = TagType.seasonal


    # --------------
    # -- Defaults --
    # --------------
    @post_processor
    def _set_defaults(self, _: object):
        self.preprocess = []
        self.state_constructor = [NukeConfig()]
