
from typing import Literal

from lib_config.config import MISSING, config

from corerl.configs.data_pipeline.imputers.per_tag.base import BasePerTagImputerConfig


@config()
class CopyImputerConfig(BasePerTagImputerConfig):
    name: Literal['copy'] = "copy"
    imputation_horizon: int = MISSING
