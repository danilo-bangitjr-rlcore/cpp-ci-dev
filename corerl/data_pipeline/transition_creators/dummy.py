from dataclasses import dataclass

from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import NewTransition, PipelineFrame
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
    def __init__(
            self,
            cfg: DummyTransitionCreatorConfig,
            tag_configs: list[TagConfig],
    ):
        super().__init__(cfg, tag_configs)

    def _inner_call(self,
                    pf: PipelineFrame,
                    tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[NewTransition], TransitionCreatorTemporalState]:
        tc_ts = TransitionCreatorTemporalState()
        return [], tc_ts


transition_creator_group.dispatcher(DummyTransitionCreator)
