from dataclasses import dataclass
from typing import Any

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group



@dataclass
class IdentityImputerConfig(BaseImputerConfig):
    name: str = "identity"


class IdentityImputer(BaseImputer):
    def __init__(self, cfg: IdentityImputerConfig, tag_cfg: Any):
        super().__init__(cfg, tag_cfg)

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf


imputer_group.dispatcher(IdentityImputer)
