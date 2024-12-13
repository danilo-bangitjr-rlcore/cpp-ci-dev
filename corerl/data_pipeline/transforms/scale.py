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
            new_name = f"({col})*{self._factor}"
            carry.transform_data[new_name] = self._factor * x
            carry.transform_data.drop(col, axis=1, inplace=True)

        return carry, None


transform_group.dispatcher(Scale)
