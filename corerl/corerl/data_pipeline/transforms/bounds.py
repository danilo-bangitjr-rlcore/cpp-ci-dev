import numpy as np

from corerl.configs.data_pipeline.transforms.bounds import BoundsConfig
from corerl.data_pipeline.transforms.base import transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


class BoundsXform:
    def __init__(self, cfg: BoundsConfig):
        self._cfg = cfg


    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            x = self._bounds_check(x)
            self._assign_to_df(carry, col, x)


        return carry, None


    def _bounds_check(self, x: np.ndarray):
        if self._cfg.mode == 'clip':
            return np.clip(x, *self._cfg.bounds)

        if self._cfg.mode == 'nan':
            return np.where((x < self._cfg.bounds[0]) | (x > self._cfg.bounds[1]), np.nan, x)

        raise NotImplementedError


    def _assign_to_df(self, carry: TransformCarry, col: str, x: np.ndarray):
        new_name = f'{col}_bounds'
        carry.transform_data.drop(col, axis=1, inplace=True)
        carry.transform_data[new_name] = x


    def reset(self):
        ...


transform_group.dispatcher(BoundsXform)
