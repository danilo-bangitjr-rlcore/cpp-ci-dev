from typing import Literal

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group


@config()
class IdentityFilterConfig(BaseOddityFilterConfig):
    name: Literal['identity'] = 'identity'


class IdentityFilter(BaseOddityFilter):
    def __init__(self, cfg: IdentityFilterConfig):
        super().__init__(cfg)

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf


outlier_group.dispatcher(IdentityFilter)
