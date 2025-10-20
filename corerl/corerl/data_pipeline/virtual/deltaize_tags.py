from collections.abc import Sequence
from functools import cached_property

from lib_defs.config_defs.tag_config import TagType
from lib_utils.list import filter_instance

from corerl.configs.data_pipeline.transforms import TransformConfig
from corerl.configs.data_pipeline.virtual.deltaize_tags import DeltaStageConfig
from corerl.configs.tags.delta import DeltaTagConfig
from corerl.configs.tags.tag_config import TagConfig
from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.state import AppState


class DeltaizeTags(Constructor):
    def __init__(self, tag_cfgs: Sequence[TagConfig], cfg: DeltaStageConfig, app_state: AppState):
        self._cfg = cfg
        self._app_state = app_state
        super().__init__(tag_cfgs)

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        transformed_parts, tag_names = self._transform_tags(pf, StageCode.VIRTUAL)
        for tag_name, transformed_part in zip(tag_names, transformed_parts, strict=True):
            pf.data[tag_name] = transformed_part

        return pf

    def _get_relevant_configs(self, tag_cfgs: Sequence[TagConfig]) -> dict[str, list[TransformConfig]]:
        return {
            tag.name: [self._cfg.delta_cfg]
            for tag in tag_cfgs
            if tag.type == TagType.delta
        }

    @cached_property
    def columns(self):
        pf = self._probe_fake_data()
        return list(pf.data.columns)

def log_delta_tags(
    app_state: AppState,
    prep_stage: Preprocessor,
    tag_cfgs: list[TagConfig],
    pf: PipelineFrame,
):
    """
    Log denormalized delta tags after outliers have been filtered and NaNs have been imputed
    """
    raw_data = prep_stage.inverse(pf.data)
    delta_cfgs = filter_instance(DeltaTagConfig, tag_cfgs)
    for tag_cfg in delta_cfgs:
        if len(raw_data[tag_cfg.name]) == 0:
            continue

        val = float(raw_data[tag_cfg.name].values[-1])
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric="DELTA-" + tag_cfg.name,
            value=val,
        )
