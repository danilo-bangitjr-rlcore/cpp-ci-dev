from typing import Literal

import numpy as np

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class ComparatorConfig(BaseTransformConfig):
    name: Literal['comparator'] = 'comparator'
    op: Literal['<', '>', '<=', '>=','==','!='] = MISSING
    val: float = 0.0

class Comparator:
    def __init__(self, cfg: ComparatorConfig):
        self._cfg = cfg
        self._op = cfg.op
        self._op_method_name = _get_op_method_name(cfg.op)
        self._val = cfg.val

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)

        for col in cols:
            x = carry.transform_data[col].to_numpy(dtype=float, na_value=np.nan)
            nan_mask = np.isnan(x)
            op_method = getattr(x, self._op_method_name)

            new_x: np.ndarray = op_method(self._val)
            new_x = new_x.astype(float) # arrays of bools don't accept nans
            new_x[nan_mask] = np.nan

            new_name = f'{col}{self._op}{self._val}'
            carry.transform_data.drop(col, axis=1, inplace=True)
            carry.transform_data[new_name] = new_x

        return carry, None

    def reset(self) -> None:
        pass

transform_group.dispatcher(Comparator)

def _get_op_method_name(op: Literal['<', '>', '<=', '>=','==','!=']) -> str:
    match op:
        case '<':
            return '__lt__'
        case '>':
            return '__gt__'
        case '<=':
            return '__le__'
        case '>=':
            return '__ge__'
        case '==':
            return '__eq__'
        case '!=':
            return '__ne__'
