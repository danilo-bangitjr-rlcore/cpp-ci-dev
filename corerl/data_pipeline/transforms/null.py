from dataclasses import dataclass

from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry
import pandas as pd


@dataclass
class NullConfig(BaseTransformConfig):
    name: str = 'null'

class Null:
    def __init__(self, cfg: NullConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        carry.transform_data = pd.DataFrame()
        return carry, None

transform_group.dispatcher(Null)
