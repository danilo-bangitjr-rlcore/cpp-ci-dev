from abc import abstractmethod
from typing import Any, Literal

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.per_tag.factory import init_per_tag_imputer
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.utils import invoke_stage_per_tag


@config()
class BaseImputerStageConfig:
    name: Any = MISSING


@config()
class PerTagImputerConfig:
    name: Literal['per-tag'] = 'per-tag'


class BaseImputer:
    def __init__(self, imputer_cfg: PerTagImputerConfig, tag_cfgs: list[TagConfig]):
        self._imputer_cfg = imputer_cfg
        self._tags = tag_cfgs

    @abstractmethod
    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        ...


class PerTagImputer(BaseImputer):
    def __init__(self, imputer_cfg: PerTagImputerConfig, tag_cfgs: list[TagConfig]):
        super().__init__(imputer_cfg, tag_cfgs)

        self._tag_imputers = {
            tag.name: init_per_tag_imputer(tag.imputer)
            for tag in tag_cfgs
        }


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        return invoke_stage_per_tag(pf, self._tag_imputers)


imputer_group = Group[
    [list[TagConfig]],
    BaseImputer
]()

def init_imputer(imputer_cfg: BaseImputerStageConfig, tag_cfgs: list[TagConfig]):
    imputer_group.dispatcher(PerTagImputer)
    return imputer_group.dispatch(imputer_cfg, tag_cfgs)
