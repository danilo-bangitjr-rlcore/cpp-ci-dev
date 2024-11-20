from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING
from corerl.utils.hydra import Group
from typing import Hashable
import numpy as np

from corerl.data_pipeline.datatypes import MissingType, PipelineFrame, update_missing_info_col
from corerl.data_pipeline.tag_config import TagConfig


@dataclass
class BaseOutlierDetectorConfig:
    name: str = MISSING


class BaseOutlierDetector(ABC):
    def __init__(self, cfg: BaseOutlierDetectorConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, pf: PipelineFrame, cfg: TagConfig) -> PipelineFrame:
        raise NotImplementedError

    def _update_missing_info(self, name: Hashable, pf: PipelineFrame, outlier_mask: np.ndarray):
        # update existing missing info
        existing_missing_mask = pf.missing_info[name] != MissingType.NULL
        update_missing_mask = existing_missing_mask & outlier_mask  # <- Series & np.ndarray results in Series
        col_name = update_missing_mask.name
        update_missing_info_col(
            missing_info=pf.missing_info,
            name=col_name,
            missing_mask=update_missing_mask.to_numpy(),
            new_val=MissingType.OUTLIER,
        )

        # add new missing info
        new_missing_mask = ~existing_missing_mask & outlier_mask
        pf.missing_info.loc[new_missing_mask, name] = MissingType.OUTLIER

outlier_group = Group[
    [], BaseOutlierDetector
]('pipeline/outlier_detector')
