from corerl.data_pipeline.datatypes import MissingType, PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig


def missing_data_checker(pf: PipelineFrame, cfg: TagConfig) -> PipelineFrame:
    missing_mask = pf.data.isna()
    pf.missing_info = pf.missing_info.mask(missing_mask, MissingType.MISSING)
    return pf
