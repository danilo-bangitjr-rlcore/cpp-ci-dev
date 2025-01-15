from corerl.models.base import BaseModel, BaseModelConfig, model_group


def init_model(cfg: BaseModelConfig) -> BaseModel:
    return model_group.dispatch(cfg)
