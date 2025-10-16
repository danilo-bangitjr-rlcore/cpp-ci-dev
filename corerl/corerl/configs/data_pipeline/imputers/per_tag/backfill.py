
from datetime import datetime
from typing import Literal

from lib_config.config import MISSING, config

from corerl.configs.data_pipeline.imputers.per_tag.base import BasePerTagImputerConfig


@config()
class BackfillImputerConfig(BasePerTagImputerConfig):
    name: Literal["backfill"] = "backfill"
    backfill_val: float = MISSING
    backfill_to: datetime = MISSING
