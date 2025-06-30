from corerl.models.base import BaseModel, BaseModelConfig, model_group
from corerl.tags.tag_config import TagConfig


def init_model(cfg: BaseModelConfig, tag_configs: list[TagConfig]) -> BaseModel:
    return model_group.dispatch(cfg, tag_configs)
