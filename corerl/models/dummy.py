from typing import Literal

from torch import Tensor

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition
from corerl.data_pipeline.tag_config import TagConfig
from corerl.models.base import BaseModel, BaseModelConfig, model_group


@config()
class DummyEndoModelConfig(BaseModelConfig):
    name: str = 'dummy'
    endogenous: bool = True


class DummyEndoModel(BaseModel):
    def __init__(self,
                 cfg: DummyEndoModelConfig,
                 tag_configs: list[TagConfig]
                 ):
        super().__init__(cfg, tag_configs)

    def fit(self, transitions: list[Transition]):
        pass

    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, float]:
        endo_obs = state[self.endo_idxs]
        return endo_obs, 0.


model_group.dispatcher(DummyEndoModel)


@config()
class DummyModelConfig(BaseModelConfig):
    name: Literal['dummy'] = 'dummy'


class DummyModel(BaseModel):
    def __init__(self,
                 cfg: DummyModelConfig,
                 tag_configs: list[TagConfig]
                 ):
        super().__init__(cfg, tag_configs)

    def fit(self, transitions: list[Transition]):
        pass

    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, float]:
        endo_obs = state
        return endo_obs, 0.


model_group.dispatcher(DummyModel)
