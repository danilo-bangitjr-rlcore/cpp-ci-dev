from dataclasses import dataclass
from typing import Literal

import numpy as np

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class DeltaConfig(BaseTransformConfig):
    name: Literal["delta"] = "delta"


@dataclass
class DeltaTemporalState:
    last: np.ndarray | None = None


class Delta:
    def __init__(self, cfg: DeltaConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        ts = ts if ts is not None else DeltaTemporalState()
        assert isinstance(ts, DeltaTemporalState)

        for i in range(len(carry.transform_data)):
            row = carry.transform_data.iloc[i].to_numpy().copy()

            if ts.last is None:
                ts.last = row
                carry.transform_data.iloc[i, :] = np.nan
                continue

            delta = row - ts.last
            ts.last = row
            carry.transform_data.iloc[i] = delta

        carry.transform_data.rename(columns=lambda col: f'{col}_delta', inplace=True)
        return carry, ts

    def reset(self) -> None:
        pass


transform_group.dispatcher(Delta)
