from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class DeltaConfig(BaseTransformConfig):
    name: Literal["delta"] = "delta"
    time_thresh: timedelta = MISSING


@dataclass
class DeltaTemporalState:
    last: np.ndarray | None = None
    time: list[datetime] | None = None


class Delta:
    def __init__(self, cfg: DeltaConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        ts = ts if ts is not None else DeltaTemporalState()
        assert isinstance(ts, DeltaTemporalState)

        for i in range(len(carry.transform_data)):
            row = carry.transform_data.iloc[i].to_numpy().copy()
            time = carry.transform_data.index[i]
            assert isinstance(time, pd.Timestamp)
            time = time.to_pydatetime()

            if ts.time is None:
                carry.transform_data.iloc[i, :] = np.nan
                ts.last = row
                ts.time = [time] * len(row)
                continue

            assert ts.time is not None
            assert ts.last is not None

            for ind in range(len(row)):
                if np.isnan(row[ind]) or (time - ts.time[ind] > self._cfg.time_thresh):
                    carry.transform_data.iloc[i, ind] = np.nan
                    if not np.isnan(row[ind]):
                        ts.last[ind] = row[ind]
                        ts.time[ind] = time
                    continue

                delta = row[ind] - ts.last[ind]
                ts.last[ind] = row[ind]
                ts.time[ind] = time
                carry.transform_data.iloc[i, ind] = delta

        carry.transform_data.rename(columns=lambda col: f'{col}_Δ', inplace=True)
        return carry, ts

    def reset(self) -> None:
        pass

    @staticmethod
    def is_delta_transformed(col: str):
        """
        Detect whether a given column has been delta transformed.

        Because the delta xform is responsible for marking a column
        as delta xformed, then the delta xform should also be
        responsible for this detection.
        """
        return '_Δ' in col

    @staticmethod
    def get_non_delta(actions: pd.DataFrame):

        direct_action_cols = [
            col for col in actions.columns
            if not Delta.is_delta_transformed(col)
        ]
        return actions[direct_action_cols]

transform_group.dispatcher(Delta)
