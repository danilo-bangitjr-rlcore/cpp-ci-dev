from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING
from corerl.utils.hydra import Group
from typing import Hashable
import numpy as np

from corerl.data_pipeline.datatypes import PipelineFrame
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


outlier_group = Group[
    [], BaseOutlierDetector
]('pipeline/outlier_detector')
