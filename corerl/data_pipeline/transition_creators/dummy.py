from dataclasses import dataclass

from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.pipeline import PipelineFrame
from corerl.data_pipeline.transition_creators.base import (
    BaseTransitionCreator,
    BaseTransitionCreatorConfig,
    transition_creator_group,
    TransitionCreatorTemporalState
)


@dataclass
class DummyTransitionCreatorConfig(BaseTransitionCreatorConfig):
    name: str = "identity"


class DummyTransitionCreator(BaseTransitionCreator):
    def __init__(self, cfg: DummyTransitionCreatorConfig):
        super().__init__(cfg)

    def _inner_call(self,
                    pf: PipelineFrame,
                    ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[Transition], TransitionCreatorTemporalState]:
        return [], ts


transition_creator_group.dispatcher(DummyTransitionCreator)
