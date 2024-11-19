from corerl.data_pipeline.outlier_detectors.base import BaseOutlierDetector
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.pipeline import TagConfig


class IdentityDetector(BaseOutlierDetector):
    def __call__(self, pf: PipelineFrame, cfg: TagConfig) -> PipelineFrame:
        raise NotImplementedError
