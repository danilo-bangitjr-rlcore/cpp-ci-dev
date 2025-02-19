from typing import Literal

from pydantic import Field

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerStageConfig
from corerl.data_pipeline.imputers.per_tag.factory import ImputerConfig, init_per_tag_imputer
from corerl.data_pipeline.imputers.per_tag.identity import IdentityImputerConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.utils import invoke_stage_per_tag


@config()
class PerTagImputerConfig(BaseImputerStageConfig):
    name: Literal['per-tag'] = 'per-tag'
    default: ImputerConfig = Field(default_factory=IdentityImputerConfig)


class PerTagImputer(BaseImputer):
    def __init__(self, imputer_cfg: PerTagImputerConfig, tag_cfgs: list[TagConfig]):
        super().__init__(imputer_cfg, tag_cfgs)

        self._tag_imputers = {
            tag.name: init_per_tag_imputer(tag.imputer)
            for tag in tag_cfgs
            if tag.imputer is not None
        }


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        return invoke_stage_per_tag(pf, self._tag_imputers)
