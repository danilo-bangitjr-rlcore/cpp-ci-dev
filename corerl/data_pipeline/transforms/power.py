from typing import Literal

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class PowerConfig(BaseTransformConfig):
    name: Literal["power"] = "power"
    pow: float = 1.0


class Power:
    def __init__(self, cfg: PowerConfig):
        self._cfg = cfg
        self._pow = cfg.pow

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            new_name = f"{col}^{self._pow}"
            carry.transform_data[new_name] = x ** self._pow
            carry.transform_data.drop(col, axis=1, inplace=True)

        return carry, None

    def reset(self):
        pass


transform_group.dispatcher(Power)
