from corerl.data_pipeline.tag_config import TagConfig
from corerl.models.base import BaseModel, BaseModelConfig, model_group


def init_model(cfg: BaseModelConfig, tag_configs: list[TagConfig]) -> BaseModel:
    return model_group.dispatch(cfg, tag_configs)
