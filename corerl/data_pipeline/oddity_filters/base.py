from abc import ABC, abstractmethod
from typing import Any

from corerl.configs.group import Group
from corerl.configs.config import config, MISSING
from corerl.data_pipeline.datatypes import PipelineFrame


@config()
class BaseOddityFilterConfig:
    name: Any = MISSING


class BaseOddityFilter(ABC):
    def __init__(self, cfg: BaseOddityFilterConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        raise NotImplementedError


outlier_group = Group[
    [], BaseOddityFilter
]()
