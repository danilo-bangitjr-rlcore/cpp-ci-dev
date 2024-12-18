from typing import Literal
from pydantic.dataclasses import dataclass as config
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.imputers.base import BaseImputer, BaseImputerConfig, imputer_group, ImputerTemporalState


@config(config={'extra': 'forbid'})
class IdentityImputerConfig(BaseImputerConfig):
    name: Literal['identity'] = "identity"


class IdentityImputer(BaseImputer):
    def __init__(self, cfg: IdentityImputerConfig):
        super().__init__(cfg)

    def _inner_call(self, pf: PipelineFrame, tag: str, imputer_ts: ImputerTemporalState | None) \
            -> tuple[PipelineFrame, ImputerTemporalState]:
        # Make sure imputer_ts is an ImputerTemporalState
        if imputer_ts is None:
            imputer_ts = ImputerTemporalState()
        assert isinstance(imputer_ts, ImputerTemporalState)

        return pf, imputer_ts


imputer_group.dispatcher(IdentityImputer)
