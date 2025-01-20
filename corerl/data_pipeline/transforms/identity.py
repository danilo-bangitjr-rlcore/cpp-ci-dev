from typing import Literal

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config(frozen=True)
class IdentityConfig(BaseTransformConfig):
    name: Literal["identity"] = "identity"


class Identity:
    def __init__(self, cfg: IdentityConfig):
        self._cfg = cfg

    def __call__(self, carry: TransformCarry, ts: object | None):
        return carry, None

    def reset(self) -> None:
        pass


transform_group.dispatcher(Identity)
