from torch import Tensor

from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition
from corerl.models.base import BaseModel, BaseModelConfig, model_group


@config()
class DummyModelConfig(BaseModelConfig):
    name: str = 'dummy'


class DummyModel(BaseModel):
    def __init__(self, cfg: DummyModelConfig):
        super().__init__(cfg)

    def fit(self, transitions: list[Transition]):
        pass

    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, float]:
        return state, 1.


model_group.dispatcher(DummyModel)
