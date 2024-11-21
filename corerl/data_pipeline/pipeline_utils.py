from corerl.data_pipeline.datatypes import Transition, PipelineFrame


def warmup_pruning(transitions: list[Transition], warmup: int) -> list[Transition]:
    return transitions[warmup:]


def handle_data_gaps(pf: PipelineFrame) -> list[PipelineFrame]:
    """
    Will split a single pipeline frame into multiple pfs where each does not have a data gap
    """
    pf.data_gap = False  # placeholder
    return [pf]
