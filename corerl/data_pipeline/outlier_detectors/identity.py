from dataclasses import dataclass

from corerl.data_pipeline.outlier_detectors.base import BaseOutlierDetector, BaseOutlierDetectorConfig, outlier_group
from corerl.data_pipeline.pipeline import PipelineFrame


@dataclass
class IdentityDetectorConfig(BaseOutlierDetectorConfig):
    name: str = 'identity'


class IdentityDetector(BaseOutlierDetector):
    def __init__(self, cfg: IdentityDetectorConfig):
        super().__init__(cfg)

    def __call__(self, pf: PipelineFrame, tag: str) -> PipelineFrame:
        return pf


outlier_group.dispatcher(IdentityDetector)
