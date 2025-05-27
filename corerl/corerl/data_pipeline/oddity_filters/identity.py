from typing import Literal

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group
from corerl.state import AppState


@config()
class IdentityFilterConfig(BaseOddityFilterConfig):
    name: Literal['identity'] = 'identity'


class IdentityFilter(BaseOddityFilter):
    def __init__(self, cfg: IdentityFilterConfig, app_state: AppState):
        super().__init__(cfg, app_state)

    def __call__(self, pf: PipelineFrame, tag: str, ts: object | None, update_stats: bool = True):
        return pf, ts


outlier_group.dispatcher(IdentityFilter)
