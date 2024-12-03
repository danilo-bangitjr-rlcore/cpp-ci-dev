from dataclasses import dataclass

from corerl.data_pipeline.oddity_filters.base import BaseOddityFilter, BaseOddityFilterConfig, outlier_group
from corerl.data_pipeline.datatypes import PipelineFrame


@dataclass
class IdentityFilterConfig(BaseOddityFilterConfig):
    name: str = 'identity'


class IdentityFilter(BaseOddityFilter):
    def __init__(self, cfg: IdentityFilterConfig):
        super().__init__(cfg)

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf


outlier_group.dispatcher(IdentityFilter)
