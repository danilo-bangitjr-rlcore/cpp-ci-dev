from abc import abstractmethod
from typing import Any

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig


@config()
class BaseImputerStageConfig:
    name: Any = MISSING


class BaseImputer:
    def __init__(self, imputer_cfg: BaseImputerStageConfig, tag_cfgs: list[TagConfig]):
        self._imputer_cfg = imputer_cfg
        self._tags = tag_cfgs

    @abstractmethod
    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ...


imputer_group = Group[
    [list[TagConfig]],
    BaseImputer
]()
