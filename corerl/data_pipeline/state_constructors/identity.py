from dataclasses import dataclass

from corerl.data_pipeline.pipeline import PipelineFrame
from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.state_constructors.base import (
    BaseStateConstructor,
    BaseStateConstructorConfig,
    state_constructor_group,
    StateConstructorTemporalState,
)


@dataclass
class IdentityStateConstructorConfig(BaseStateConstructorConfig):
    name: str = "identity"


class IdentityStateConstructor(BaseStateConstructor):
    def __init__(self, cfg: IdentityStateConstructorConfig):
        super().__init__(cfg)

    def _inner_call(self,
                    pf: PipelineFrame,
                    sc_ts: StateConstructorTemporalState | None) \
        -> tuple[list[Transition], StateConstructorTemporalState]:
        sc_ts =  StateConstructorTemporalState()
        return pf.transitions, sc_ts

    def reset(self) -> None:
        pass


state_constructor_group.dispatcher(IdentityStateConstructor)
