from abc import ABC, abstractmethod

from lib_config.group import Group

from corerl.configs.data_pipeline.oddity_filters.base import BaseOddityFilterConfig
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.state import AppState


class BaseOddityFilter(ABC):
    def __init__(self, cfg: BaseOddityFilterConfig, app_state: AppState):
        self.cfg = cfg
        self._app_state = app_state

    @abstractmethod
    def __call__(self, pf: PipelineFrame, tag: str, ts: object | None) -> tuple[PipelineFrame, object | None]:
        raise NotImplementedError


outlier_group = Group[
    [AppState], BaseOddityFilter,
]()
