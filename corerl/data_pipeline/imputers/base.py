from dataclasses import dataclass
from abc import ABC, abstractmethod

from corerl.data.data import PipelineFrame
from corerl.data_pipeline.pipeline import TagConfig


@dataclass
class ImputerConfig:
    pass


class BaseImputer(ABC):
    @abstractmethod
    def __call__(self, pf: PipelineFrame, cfg: TagConfig) -> PipelineFrame:
        raise NotImplementedError
