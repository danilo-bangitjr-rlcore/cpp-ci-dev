from dataclasses import dataclass

from corerl.data_pipeline.transforms import ProductConfig
from corerl.data_pipeline.transforms.base import transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@dataclass
class ProductTemporalState:
    other_ts: object | None = None


class ProductTransform:
    def __init__(self, cfg: ProductConfig):
        self._cfg = cfg
        self._other = cfg.other
        self._other_xform = [transform_group.dispatch(xform) for xform in cfg.other_xform]

    def __call__(self, carry: TransformCarry, ts: object | None):
        assert isinstance(ts, ProductTemporalState | None)
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

        # take product with other
        cols = set(carry.transform_data.columns)
        other_name = other_carry.transform_data.columns[0]
        other_vals = other_carry.transform_data[other_name]
        for col in cols:
            new_name = f"({col})*({other_name})"
            carry.transform_data[new_name] = carry.transform_data[col] * other_vals
            carry.transform_data.drop(col, axis=1, inplace=True)

        return carry, ProductTemporalState(other_ts=other_ts)

    def reset(self) -> None:
        for xform in self._other_xform:
            xform.reset()


transform_group.dispatcher(ProductTransform)
