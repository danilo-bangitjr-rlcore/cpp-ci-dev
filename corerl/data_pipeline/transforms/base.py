from dataclasses import dataclass
from typing import Protocol

from omegaconf import MISSING
from corerl.utils.hydra import Group
from corerl.data_pipeline.transforms.interface import TransformCarry

class Transform(Protocol):
    def __call__(self, carry: TransformCarry, ts: object | None) -> tuple[TransformCarry, object | None]: ...

@dataclass
class BaseTransformConfig:
    name: str = MISSING

sc_group = Group[
    [], Transform,
# As far as I am aware, there is no way to do group overrides
# in a nested list of lists of groups.
]('sc')
