from typing import Literal

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.per_tag.base import BasePerTagImputer, BasePerTagImputerConfig, per_tag_imputer_group


@config()
class IdentityImputerConfig(BasePerTagImputerConfig):
    name: Literal['identity'] = "identity"


class IdentityImputer(BasePerTagImputer):
    def __init__(self, cfg: IdentityImputerConfig):
        super().__init__(cfg)


    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf


per_tag_imputer_group.dispatcher(IdentityImputer)
