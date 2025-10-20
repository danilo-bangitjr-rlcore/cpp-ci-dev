from corerl.configs.data_pipeline.imputers.per_tag.identity import IdentityImputerConfig
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.per_tag.base import BasePerTagImputer, per_tag_imputer_group


class IdentityImputer(BasePerTagImputer):
    def __init__(self, cfg: IdentityImputerConfig):
        super().__init__(cfg)


    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf


per_tag_imputer_group.dispatcher(IdentityImputer)
