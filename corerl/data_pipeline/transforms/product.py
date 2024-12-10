from dataclasses import dataclass, field

from omegaconf import MISSING

from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.identity import IdentityConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.utils.hydra import list_


@dataclass
class ProductConfig(BaseTransformConfig):
    name: str = "product"

    other: str = MISSING
    other_xform: list[BaseTransformConfig] = list_([IdentityConfig])


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
        assert other_data is not None

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
        carry.transform_data = carry.transform_data.mul(other_carry.transform_data.values, axis="index")

        return carry, ProductTemporalState(other_ts=other_ts)


transform_group.dispatcher(ProductTransform)
