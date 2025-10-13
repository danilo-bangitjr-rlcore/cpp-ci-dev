
from typing import Literal

from lib_config.config import config

from corerl.configs.data_pipeline.oddity_filters.base import BaseOddityFilterConfig


@config()
class StuckDetectorConfig(BaseOddityFilterConfig):
    name: Literal["stuck_detector"] = "stuck_detector"
    eps: float = 1e-3
    step_tol: int = 10  #  number of steps before marking stuck
