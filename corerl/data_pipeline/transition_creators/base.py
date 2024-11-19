from dataclasses import dataclass
from abc import ABC, abstractmethod

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.pipeline import TagConfig
from corerl.data_pipeline.datatypes import Transition


@dataclass
class BaseTransitionCreatorConfig:
    pass


class BaseTransitionCreator(ABC):
    @abstractmethod
    def __call__(self, pf: PipelineFrame, cfg: TagConfig) -> list[Transition]:
        raise NotImplementedError
