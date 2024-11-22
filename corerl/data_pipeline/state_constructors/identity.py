from dataclasses import dataclass

from corerl.data_pipeline.datatypes import Transition, PipelineFrame
from corerl.data_pipeline.state_constructors.base import (
    BaseStateConstructor,
    BaseStateConstructorConfig,
    state_constructor_group,
)


@dataclass
class IdentityStateConstructorConfig(BaseStateConstructorConfig):
    name: str = "identity"


class IdentityStateConstructor(BaseStateConstructor):
    def __init__(self, cfg: IdentityStateConstructorConfig):
        super().__init__(cfg)

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf

    def reset(self) -> None:
        pass


state_constructor_group.dispatcher(IdentityStateConstructor)
