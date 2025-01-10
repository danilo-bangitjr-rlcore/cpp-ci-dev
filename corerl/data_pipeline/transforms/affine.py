from typing import Literal

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config(frozen=True)
class AffineConfig(BaseTransformConfig):
    name: Literal["affine"] = "affine"
    scale: float = 1.0
    bias: float = 0.0


class Affine:
    def __init__(self, cfg: AffineConfig):
        self._cfg = cfg
        self._scale = cfg.scale
        self._bias = cfg.bias

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            new_name = f"{self._scale}*{col}+{self._bias}"
            carry.transform_data[new_name] = self._scale * x + self._bias
            carry.transform_data.drop(col, axis=1, inplace=True)

        return carry, None

    def reset(self) -> None:
        pass


transform_group.dispatcher(Affine)
