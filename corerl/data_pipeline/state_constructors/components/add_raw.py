from dataclasses import dataclass

from corerl.data_pipeline.state_constructors.components.base import BaseTransformConfig, sc_group
from corerl.data_pipeline.state_constructors.interface import TransformCarry


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

sc_group.dispatcher(AddRaw)
