from abc import abstractmethod

from lib_config.group import Group

from corerl.configs.data_pipeline.imputers.per_tag.base import BasePerTagImputerConfig
from corerl.data_pipeline.datatypes import PipelineFrame


class BasePerTagImputer:
    def __init__(self, cfg: BasePerTagImputerConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        ...


per_tag_imputer_group = Group[
    [], BasePerTagImputer,
]()
