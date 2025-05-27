import logging
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.per_tag.base import BasePerTagImputer, BasePerTagImputerConfig, per_tag_imputer_group

logger = logging.getLogger(__name__)

@config()
class BackfillImputerConfig(BasePerTagImputerConfig):
    name: Literal["backfill"] = "backfill"
    backfill_val: float = MISSING
    backfill_to: datetime = MISSING


class BackfillImputer(BasePerTagImputer):
    def __init__(self, cfg: BackfillImputerConfig):
        super().__init__(cfg)
        self.backfill_val = cfg.backfill_val
        self.backfill_to = cfg.backfill_to

    def __call__(self, pf: PipelineFrame, tag: str):
        idx = pf.data.index
        if not isinstance(idx, pd.DatetimeIndex):
            logger.warning("Backfill imputer called on df without a datetime index.")
            return pf

        time = idx.to_pydatetime()
        tag_data = pf.data[tag].to_numpy()
        tag_data = np.where(time <= self.backfill_to, self.backfill_val, tag_data)

        pf.data[tag] = tag_data
        return pf


per_tag_imputer_group.dispatcher(BackfillImputer)
