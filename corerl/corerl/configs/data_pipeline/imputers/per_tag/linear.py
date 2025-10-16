
from typing import Literal

from lib_config.config import MISSING
from pydantic.dataclasses import dataclass as config

from corerl.configs.data_pipeline.imputers.per_tag.base import BasePerTagImputerConfig


@config(config={'extra': 'forbid'})
class LinearImputerConfig(BasePerTagImputerConfig):
    name: Literal['linear'] = "linear"
    max_gap: int = MISSING  # Maximum number of NaNs between the two values used in linear interpolation
