from typing import Literal
import pandas as pd

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config(frozen=True)
class NullConfig(BaseTransformConfig):
    name: Literal['null'] = 'null'


class Null:
    def __init__(self, cfg: NullConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        carry.transform_data = pd.DataFrame()
        return carry, None

transform_group.dispatcher(Null)
