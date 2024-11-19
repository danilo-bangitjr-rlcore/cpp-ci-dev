from corerl.data.data import PipelineFrame
from corerl.data_pipeline.pipeline import TagConfig
from corerl.data_pipeline.imputers.base import BaseImputer


class IdentityImputer(BaseImputer):
    def __call__(self, pf: PipelineFrame, cfg: TagConfig) -> PipelineFrame:
        return pf
