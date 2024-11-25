from dataclasses import dataclass

from corerl.data_pipeline.datatypes import Transition, PipelineFrame
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
                    tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[Transition], TransitionCreatorTemporalState]:
        tc_ts = TransitionCreatorTemporalState()
        return [], tc_ts


transition_creator_group.dispatcher(DummyTransitionCreator)
