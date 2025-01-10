from typing import Literal

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config(frozen=True)
class AddRawConfig(BaseTransformConfig):
    name: Literal['add_raw'] = 'add_raw'

class AddRaw:
    def __init__(self, cfg: AddRawConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        raw_obs = carry.obs[carry.tag]

        if carry.tag not in carry.transform_data:
            carry.transform_data[carry.tag] = raw_obs

        return carry, None
        
    def reset(self) -> None:
        pass

transform_group.dispatcher(AddRaw)
