from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING

from corerl.utils.hydra import Group
from corerl.data_pipeline.datatypes import Transition, StageCode, StageTemporalState, PipelineFrame


@dataclass
class BaseTransitionCreatorConfig:
    name: str = MISSING


@dataclass
class TransitionCreatorTemporalState(StageTemporalState):
    pass


class BaseTransitionCreator(ABC):
    def __init__(self, cfg: BaseTransitionCreatorConfig):
        self.cfg = cfg
        self.stage_code = StageCode.TC

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        """
        Gets the current temporal state
        """
        tc_ts: TransitionCreatorTemporalState | None = pf.temporal_state.get_ts(self.stage_code)
        transitions, new_tc_ts = self._inner_call(pf, tc_ts)
        pf.temporal_state = pf.temporal_state.update(new_tc_ts, self.stage_code)
        pf.transitions = transitions
        return pf

    @abstractmethod
    def _inner_call(self, pf: PipelineFrame, tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[Transition], TransitionCreatorTemporalState]:
        raise NotImplementedError


transition_creator_group = Group[
    [], BaseTransitionCreator
]('pipeline/transition_creator')
