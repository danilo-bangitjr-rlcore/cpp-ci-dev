from typing import Any
from torch import Tensor

from corerl.data_pipeline.datatypes import Transition
from corerl.configs.group import Group
from corerl.configs.config import config, MISSING


@config()
class BaseModelConfig:
    name: Any = MISSING


class BaseModel:
    def __init__(self, cfg: BaseModelConfig):
        self.cfg = cfg

    def fit(self, transitions: list[Transition]):
        ...

    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, float]:
        ...


model_group = Group[
    [], BaseModelConfig
]()
