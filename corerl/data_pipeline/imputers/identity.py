from typing import Literal

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group


@config()
class IdentityImputerConfig(BaseImputerConfig):
    name: Literal['identity'] = "identity"


class IdentityImputer(BaseImputer):
    def __init__(self, cfg: IdentityImputerConfig):
        super().__init__(cfg)


    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf


imputer_group.dispatcher(IdentityImputer)
