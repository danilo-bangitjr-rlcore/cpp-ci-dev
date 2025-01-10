from typing import Literal
from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config(frozen=True)
class ScaleConfig(BaseTransformConfig):
    name: Literal['scale'] = "scale"
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

    def reset(self) -> None:
        pass


transform_group.dispatcher(Scale)
