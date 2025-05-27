from corerl.data_pipeline.datatypes import MissingType, PipelineFrame


def missing_data_checker(pf: PipelineFrame, tag: str) -> PipelineFrame:
    missing_mask = pf.data[tag].isna()
    pf.missing_info[tag] = pf.missing_info[tag].mask(missing_mask, MissingType.MISSING)
    return pf
