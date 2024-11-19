from corerl.data.data import PipelineFrame
from corerl.data_pipeline.pipeline import TagConfig


def identity_bound_checker(pf: PipelineFrame, cfg: TagConfig) -> PipelineFrame:
    return pf
