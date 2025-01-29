from typing import Literal

import numpy as np

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class InverseConfig(BaseTransformConfig):
    name: Literal["inverse"] = "inverse"
    tolerance: float = 1e-4

class Inverse:
    def __init__(self, cfg: InverseConfig):
        self._cfg = cfg
        self._tol = cfg.tolerance

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            nonzero = np.abs(x) > self._tol
            new_x = np.full_like(x, np.nan)
            new_x[nonzero] = 1 / x[nonzero]
            self._assign_to_df(carry, col, new_x)

        return carry, None

    def _assign_to_df(self, carry: TransformCarry, col: str, x: np.ndarray):
        new_name = f"1/{col}"
        carry.transform_data.drop(col, axis=1, inplace=True)
        carry.transform_data[new_name] = x

    def reset(self) -> None:
        pass


transform_group.dispatcher(Inverse)
