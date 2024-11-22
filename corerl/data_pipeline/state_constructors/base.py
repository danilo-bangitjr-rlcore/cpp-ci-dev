from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING

from corerl.utils.hydra import Group
from corerl.data_pipeline.datatypes import Transition, StageCode, StageTemporalState, PipelineFrame

@dataclass
class BaseStateConstructorConfig:
    name: str = MISSING


@dataclass
class StateConstructorTemporalState(StageTemporalState):
    pass


class BaseStateConstructor(ABC):
    def __init__(self, cfg: BaseStateConstructorConfig):
        self.cfg = cfg
        self.stage_code = StageCode.SC

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        sc_ts: StateConstructorTemporalState | None = pf.temporal_state.get_ts(self.stage_code, tag)
        transitions_with_state, new_sc_ts = self._inner_call(pf, sc_ts)
        print(pf.temporal_state)

        pf.temporal_state = pf.temporal_state.update(new_sc_ts, self.stage_code, tag)
        pf.transitions = transitions_with_state
        return pf

    @abstractmethod
    def _inner_call(self, pf: PipelineFrame, sc_ts: StateConstructorTemporalState | None) \
            -> tuple[list[Transition], StateConstructorTemporalState]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


state_constructor_group = Group[
    [], BaseStateConstructor
]('pipeline/state_constructor')
