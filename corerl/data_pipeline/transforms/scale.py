from dataclasses import dataclass

from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@dataclass
class ScaleConfig(BaseTransformConfig):
    name: str = "scale"
    factor: float = 1.0


class Scale:
    def __init__(self, cfg: ScaleConfig):
        self._cfg = cfg
        self._factor = cfg.factor

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            carry.transform_data[col] = self._factor * x

        return carry, None


transform_group.dispatcher(Scale)
