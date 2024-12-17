from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

from corerl.configs.config import config
from corerl.configs.group import Group
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import NewTransition, StageCode, PipelineFrame

@config(frozen=True)
class BaseTransitionCreatorConfig:
    name: Any = 'base'


@dataclass
class TransitionCreatorTemporalState:
    pass


class BaseTransitionCreator(ABC):
    def __init__(
            self,
            cfg: BaseTransitionCreatorConfig,
            tag_configs: list[TagConfig]
    ):
        self.cfg = cfg
        self.stage_code = StageCode.TC
        self.tag_configs = tag_configs
        self._init_action_tags()

    def _init_action_tags(self):
        self.action_tags = []
        for tag_config in self.tag_configs:
            name = tag_config.name
            if tag_config.is_action:
                self.action_tags.append(name)

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        tc_ts = pf.temporal_state.get(self.stage_code)
        assert isinstance(tc_ts, TransitionCreatorTemporalState | None)
        transitions, new_tc_ts = self._inner_call(pf, tc_ts)
        pf.temporal_state[self.stage_code] = new_tc_ts
        pf.transitions = transitions
        return pf

    @abstractmethod
    def _inner_call(self, pf: PipelineFrame, tc_ts: TransitionCreatorTemporalState | None) \
            -> tuple[list[NewTransition], TransitionCreatorTemporalState | None]:
        raise NotImplementedError

transition_creator_group = Group[
    [list[TagConfig]], BaseTransitionCreator
]()
