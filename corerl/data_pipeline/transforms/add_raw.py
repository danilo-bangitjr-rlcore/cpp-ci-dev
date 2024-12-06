from dataclasses import dataclass

from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@dataclass
class AddRawConfig(BaseTransformConfig):
    name: str = 'add_raw'

class AddRaw:
    def __init__(self, cfg: AddRawConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        raw_obs = carry.obs[carry.tag]

        if carry.tag not in carry.transform_data:
            carry.transform_data[carry.tag] = raw_obs

        return carry, None

transform_group.dispatcher(AddRaw)
