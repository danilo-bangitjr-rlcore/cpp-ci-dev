from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING
from corerl.utils.hydra import Group

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.pipeline import TagConfig
from corerl.data_pipeline.datatypes import Transition


@dataclass
class BaseTransitionCreatorConfig:
    name: str = MISSING


class BaseTransitionCreator(ABC):
    def __init__(self, cfg: BaseTransitionCreatorConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, pf: PipelineFrame, cfg: TagConfig) -> list[Transition]:
        raise NotImplementedError


transition_creator_group = Group[
    [], BaseTransitionCreatorConfig
]('pipeline/transition_creator')
