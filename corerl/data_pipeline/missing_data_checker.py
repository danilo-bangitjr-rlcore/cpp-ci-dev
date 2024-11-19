from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig


def missing_data_checker(pf: PipelineFrame, cfg: TagConfig) -> PipelineFrame:
    return pf
