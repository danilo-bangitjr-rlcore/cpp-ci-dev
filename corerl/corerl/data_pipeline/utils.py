from collections.abc import Callable, Mapping

from corerl.data_pipeline.datatypes import PipelineStage, StageCode, TagName


def invoke_stage_per_tag[T](carry: T, stage: Mapping[TagName, PipelineStage[T]]) -> T:
    for tag, f in stage.items():
        carry = f(carry, tag)

    return carry


def get_tag_temporal_state[T](
        stage: StageCode,
        tag: str,
        ts: dict[StageCode, object | None],
        default: Callable[[], T],
    ) -> T:
    # if this stage does not have a state on the ts
    # create it and attach to the ts
    stage_ts = ts.get(stage, {})
    ts[stage] = stage_ts
    assert isinstance(stage_ts, dict)

    # if a tag_ts does not exist, create it and attach
    # it to the stage_ts
    tag_ts = stage_ts.get(tag, default())
    stage_ts[tag] = tag_ts

    return tag_ts
