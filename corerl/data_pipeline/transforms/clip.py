from typing import Literal

import numpy as np

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class ClipConfig(BaseTransformConfig):
    name: Literal['clip'] = 'clip'
    bounds: tuple[float, float] = MISSING


class Clip:
    def __init__(self, cfg: ClipConfig):
        self._cfg = cfg


    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            x = np.clip(x, *self._cfg.bounds)
            self._assign_to_df(carry, col, x)


        return carry, None


    def _assign_to_df(self, carry: TransformCarry, col: str, x: np.ndarray):
        new_name = f'{col}_clip'
        carry.transform_data.drop(col, axis=1, inplace=True)
        carry.transform_data[new_name] = x


    def reset(self):
        ...


transform_group.dispatcher(Clip)
