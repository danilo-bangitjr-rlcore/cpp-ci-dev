from typing import Any
from abc import abstractmethod

from corerl.configs.group import Group
from corerl.configs.config import config, MISSING
from corerl.data_pipeline.datatypes import PipelineFrame


@config()
class BaseImputerConfig:
    name: Any = MISSING


class BaseImputer:
    def __init__(self, cfg: BaseImputerConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        ...


imputer_group = Group[
    [], BaseImputer
]()
