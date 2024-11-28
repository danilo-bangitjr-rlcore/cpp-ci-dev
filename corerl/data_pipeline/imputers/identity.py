from dataclasses import dataclass
from typing import Any

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group, ImputerTemporalState



@dataclass
class IdentityImputerConfig(BaseImputerConfig):
    name: str = "identity"


class IdentityImputer(BaseImputer):
    def __init__(self, cfg: IdentityImputerConfig, tag_cfg: Any):
        super().__init__(cfg, tag_cfg)

    def _inner_call(self, pf: PipelineFrame, tag: str, imputer_ts: ImputerTemporalState | None) \
            -> tuple[PipelineFrame, ImputerTemporalState]:
        # Make sure imputer_ts is an ImputerTemporalState
        if imputer_ts is None:
            imputer_ts = ImputerTemporalState()
        assert isinstance(imputer_ts, ImputerTemporalState)

        return pf, imputer_ts


imputer_group.dispatcher(IdentityImputer)
