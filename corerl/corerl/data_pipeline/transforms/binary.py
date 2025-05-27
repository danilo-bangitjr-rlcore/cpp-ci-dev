from dataclasses import dataclass

import numpy as np

from corerl.data_pipeline.transforms import BinaryConfig
from corerl.data_pipeline.transforms.base import transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@dataclass
class BinaryTemporalState:
    other_ts: object | None = None


class BinaryTransform:
    def __init__(self, cfg: BinaryConfig):
        self._cfg = cfg
        self._op = cfg.op
        self._other = cfg.other
        self._other_xform = [transform_group.dispatch(xform) for xform in cfg.other_xform]

    def __call__(self, carry: TransformCarry, ts: object | None):
        assert isinstance(ts, BinaryTemporalState | None)
        other_ts = None if ts is None else ts.other_ts

        # get data from "other" column and create carry object
        other_data = carry.obs.get([self._other])
        assert other_data is not None, f"carry obs cols: {carry.obs.columns}, other name: {self._other}"

        other_carry = TransformCarry(
            obs=carry.obs,
            transform_data=other_data.copy(),
            tag=self._other,
        )

        # execute other transform
        for xform in self._other_xform:
            other_carry, other_ts = xform(other_carry, other_ts)

        assert len(other_carry.transform_data.columns) == 1

        # apply binary op with other
        cols = set(carry.transform_data.columns)
        other_name = other_carry.transform_data.columns[0]
        for col in cols:
            self.apply_op(carry, other_carry, col, str(other_name))
            carry.transform_data.drop(col, axis=1, inplace=True)

        return carry, BinaryTemporalState(other_ts=other_ts)

    def apply_op(self, carry: TransformCarry, other_carry: TransformCarry, col: str, other_name: str) -> None:
        """
        adds col corresponding to op(col, other)
        """
        other_vals = other_carry.transform_data[other_name]
        match self._op:
            case "prod":
                new_name = f"({col})*({other_name})"
                carry.transform_data[new_name] = carry.transform_data[col] * other_vals
            case "max":
                new_name = f"max({col}, {other_name})"
                carry.transform_data[new_name] = np.maximum(carry.transform_data[col], other_vals)
            case "min":
                new_name = f"min({col}, {other_name})"
                carry.transform_data[new_name] = np.minimum(carry.transform_data[col], other_vals)
            case "add":
                new_name = f"({col}) + ({other_name})"
                carry.transform_data[new_name] = carry.transform_data[col] + other_vals
            case "replace":
                new_name = other_name
                carry.transform_data[new_name] = other_vals

    def reset(self) -> None:
        for xform in self._other_xform:
            xform.reset()


transform_group.dispatcher(BinaryTransform)
