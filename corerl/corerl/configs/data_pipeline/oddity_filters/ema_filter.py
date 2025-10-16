
from typing import Literal

from lib_config.config import config

from corerl.configs.data_pipeline.oddity_filters.base import BaseOddityFilterConfig


@config()
class EMAFilterConfig(BaseOddityFilterConfig):
    name: Literal["exp_moving"] = "exp_moving"
    alpha: float = 0.99
    tolerance: float = 2.0
    warmup: int = 10  #  number of warmup steps before rejecting
