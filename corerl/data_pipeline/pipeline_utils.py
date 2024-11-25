from corerl.data_pipeline.datatypes import PipelineFrame


def warmup_pruning(pf: PipelineFrame, warmup: int) -> PipelineFrame:
    assert pf.transitions is not None
    pf.transitions = pf.transitions[warmup:]
    return pf


def handle_data_gaps(pf: PipelineFrame) -> list[PipelineFrame]:
    """
    Will split a single pipeline frame into multiple pfs where each does not have a data gap
    """
    pf.data_gap = False  # placeholder
    return [pf]
