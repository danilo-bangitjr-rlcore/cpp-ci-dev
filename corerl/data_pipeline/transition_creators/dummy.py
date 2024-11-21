from dataclasses import dataclass

from corerl.data_pipeline.datatypes import PipelineFrame, Transition
from corerl.data_pipeline.transition_creators.base import (
    BaseTransitionCreator,
    BaseTransitionCreatorConfig,
    transition_creator_group,
)


@dataclass
class DummyTransitionCreatorConfig(BaseTransitionCreatorConfig):
    name: str = "identity"


class DummyTransitionCreator(BaseTransitionCreator):
    def __init__(self, cfg: DummyTransitionCreatorConfig):
        super().__init__(cfg)

    def __call__(self, pf: PipelineFrame) -> list[Transition]:
        return []


transition_creator_group.dispatcher(DummyTransitionCreator)
