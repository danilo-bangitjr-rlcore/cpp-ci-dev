from dataclasses import dataclass

from corerl.data_pipeline.transforms.base import BaseTransformConfig, sc_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@dataclass
class IdentityConfig(BaseTransformConfig):
    name: str = "identity"


class Identity:
    def __init__(self, cfg: IdentityConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        return carry, None


sc_group.dispatcher(Identity)
