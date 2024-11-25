from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING

from corerl.utils.hydra import Group
from corerl.data_pipeline.datatypes import PipelineFrame


@dataclass
class BaseImputerConfig:
    name: str = MISSING


class BaseImputer(ABC):
    def __init__(self, cfg: BaseImputerConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        raise NotImplementedError


imputer_group = Group[
    [], BaseImputer
]('pipeline/imputer')
