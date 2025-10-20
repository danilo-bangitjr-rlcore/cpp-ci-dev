
from __future__ import annotations

from lib_config.config import config, list_

from corerl.configs.data_pipeline.oddity_filters import OddityFilterConfig


@config()
class GlobalOddityFilterConfig:
    defaults: list[OddityFilterConfig] = list_([])
