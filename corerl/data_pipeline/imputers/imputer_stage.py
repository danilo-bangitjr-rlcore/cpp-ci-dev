from typing import Any

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.factory import init_imputer
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.utils import invoke_stage_per_tag


@config()
class BaseImputerStageConfig:
    name: Any = MISSING


class Imputer:
    def __init__(self, imputer_cfg: BaseImputerStageConfig, tag_cfgs: list[TagConfig]):
        self._imputer_cfg = imputer_cfg
        self._tags = tag_cfgs

        self._tag_imputers = {
            tag.name: init_imputer(tag.imputer)
            for tag in tag_cfgs
        }


    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        return invoke_stage_per_tag(pf, self._tag_imputers)
