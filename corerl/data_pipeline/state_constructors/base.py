from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import MISSING

from corerl.utils.hydra import Group
from corerl.data_pipeline.datatypes import Transition, StageCode, PipelineFrame

@dataclass
class BaseStateConstructorConfig:
    name: str = MISSING


@dataclass
class StateConstructorTemporalState:
    pass

from typing import Dict, cast


class BaseStateConstructor(ABC):
    def __init__(self, cfg: BaseStateConstructorConfig):
        self.cfg = cfg
        self.stage_code = StageCode.SC

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        assert pf.temporal_state is not None
        assert isinstance(pf.temporal_state, dict)

        return_val = pf.temporal_state.get(self.stage_code)
        if return_val is None:
            pf.temporal_state[self.stage_code] = dict()

        stage_dict = pf.temporal_state[self.stage_code]
        assert isinstance(stage_dict, dict)
        stage_dict = cast(
            Dict[str, StateConstructorTemporalState],
            stage_dict
        )

        sc_ts = stage_dict.get(tag)
        transitions_with_state, new_sc_ts = self._inner_call(pf, sc_ts)
        stage_dict[tag] = new_sc_ts
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
