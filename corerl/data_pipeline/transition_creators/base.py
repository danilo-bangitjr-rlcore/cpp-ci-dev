from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING

from corerl.utils.hydra import Group
from corerl.data_pipeline.datatypes import Transition, StageCode, PipelineFrame


@dataclass
class BaseTransitionCreatorConfig:
    name: str = MISSING


@dataclass
class TransitionCreatorTemporalState:
    pass


class BaseTransitionCreator(ABC):
    def __init__(self, cfg: BaseTransitionCreatorConfig):
        self.cfg = cfg
        self.stage_code = StageCode.TC

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        assert pf.temporal_state is not None
        assert isinstance(pf.temporal_state, dict)
        tc_ts = pf.temporal_state.get(self.stage_code)
        assert isinstance(tc_ts, TransitionCreatorTemporalState) or tc_ts is None
        transitions, new_tc_ts = self._inner_call(pf, tc_ts)
        pf.temporal_state[self.stage_code] = new_tc_ts
        pf.transitions = transitions
        return pf

    @abstractmethod
    def _inner_call(self, pf: PipelineFrame, tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[Transition], TransitionCreatorTemporalState]:
        raise NotImplementedError


transition_creator_group = Group[
    [], BaseTransitionCreator
]('pipeline/transition_creator')
