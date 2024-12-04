from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING
from corerl.utils.hydra import Group

from corerl.data_pipeline.datatypes import PipelineFrame


@dataclass
class BaseOddityFilterConfig:
    name: str = MISSING


class BaseOddityFilter(ABC):
    def __init__(self, cfg: BaseOddityFilterConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        raise NotImplementedError


outlier_group = Group[
    [], BaseOddityFilter
]('pipeline/outlier_detector')
