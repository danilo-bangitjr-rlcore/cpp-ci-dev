from typing import Protocol, runtime_checkable

import numpy as np
from lib_config.group import Group

from corerl.data_pipeline.transforms.interface import TransformCarry


class Transform(Protocol):
    def __call__(self, carry: TransformCarry, ts: object | None) -> tuple[TransformCarry, object | None]: ...
    def reset(self) -> None: ...


@runtime_checkable
class InvertibleTransform(Protocol):
    def invert(self, x: np.ndarray, col: str) -> np.ndarray: ...


transform_group = Group[
    [], Transform,
# As far as I am aware, there is no way to do group overrides
# in a nested list of lists of groups.
]()
