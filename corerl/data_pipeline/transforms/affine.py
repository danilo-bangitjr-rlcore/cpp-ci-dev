
from dataclasses import dataclass

from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@dataclass
class AffineConfig(BaseTransformConfig):
    name: str = "affine"
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
            carry.transform_data[col] = self._scale * x + self._bias

        return carry, None


transform_group.dispatcher(Affine)
