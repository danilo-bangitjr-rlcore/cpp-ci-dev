from dataclasses import dataclass


from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.pipeline import TagConfig
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group

@dataclass
class IdentityImputerConfig(BaseImputerConfig):
    name: str = 'identity'


class IdentityImputer(BaseImputer):
    def __call__(self, pf: PipelineFrame, cfg: TagConfig) -> PipelineFrame:
        return pf

imputer_group.dispatcher(IdentityImputer)
