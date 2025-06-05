from pydantic import Field

from corerl.configs.config import config
from corerl.data_pipeline.constructors.constructor import Constructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.tag_config import TagConfig, TagType
from corerl.data_pipeline.transforms import TransformConfig
from corerl.data_pipeline.transforms.delta import DeltaConfig


@config()
class DeltaStageConfig:
    delta_cfg: DeltaConfig = Field(default_factory=DeltaConfig)

class DeltaizeTags(Constructor):
    def __init__(self, tag_cfgs: list[TagConfig], cfg: DeltaStageConfig):
        self._cfg = cfg
        super().__init__(tag_cfgs)

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        transformed_parts, tag_names = self._transform_tags(pf, StageCode.DELTA)
        for tag_name, transformed_part in zip(tag_names, transformed_parts, strict=True):
            pf.data[tag_name] = transformed_part

        return pf

    def _get_relevant_configs(self, tag_cfgs: list[TagConfig]) -> dict[str, list[TransformConfig]]:
        return {
            tag.name: [self._cfg.delta_cfg]
            for tag in tag_cfgs
            if tag.type == TagType.delta
        }

