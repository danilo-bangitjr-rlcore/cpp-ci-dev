from abc import ABC, abstractmethod
from typing import Any

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.state import AppState


@config()
class BaseOddityFilterConfig:
    name: Any = MISSING


class BaseOddityFilter(ABC):
    def __init__(self, cfg: BaseOddityFilterConfig, app_state: AppState):
        self.cfg = cfg
        self._app_state = app_state

    @abstractmethod
    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        raise NotImplementedError


outlier_group = Group[
    [AppState], BaseOddityFilter
]()
