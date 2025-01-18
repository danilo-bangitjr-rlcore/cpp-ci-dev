from typing import Any, Protocol

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group
from corerl.data_pipeline.transforms.interface import TransformCarry


class Transform(Protocol):
    def __call__(self, carry: TransformCarry, ts: object | None) -> tuple[TransformCarry, object | None]: ...
    def reset(self) -> None: ...


@config()
class BaseTransformConfig:
    name: Any = MISSING

transform_group = Group[
    [], Transform,
# As far as I am aware, there is no way to do group overrides
# in a nested list of lists of groups.
]()
