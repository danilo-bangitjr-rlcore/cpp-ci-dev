from typing import Any, Protocol

from corerl.configs.group import Group
from corerl.configs.config import config, MISSING
from corerl.data_pipeline.transforms.interface import TransformCarry

class Transform(Protocol):
    def __call__(self, carry: TransformCarry, ts: object | None) -> tuple[TransformCarry, object | None]: ...
    def reset(self) -> None: ...


@config(frozen=True)
class BaseTransformConfig:
    name: Any = MISSING

transform_group = Group[
    [], Transform,
# As far as I am aware, there is no way to do group overrides
# in a nested list of lists of groups.
]()
