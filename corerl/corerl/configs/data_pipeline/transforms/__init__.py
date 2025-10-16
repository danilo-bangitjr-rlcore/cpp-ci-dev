from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from lib_config.config import MISSING, config, list_, post_processor
from pydantic import Field

from corerl.configs.data_pipeline.transforms.add_raw import AddRawConfig
from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig
from corerl.configs.data_pipeline.transforms.bounds import BoundsConfig
from corerl.configs.data_pipeline.transforms.delta import DeltaConfig
from corerl.configs.data_pipeline.transforms.identity import IdentityConfig
from corerl.configs.data_pipeline.transforms.norm import NormalizerConfig
from corerl.configs.data_pipeline.transforms.nuke import NukeConfig
from corerl.configs.data_pipeline.transforms.trace import TraceConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class SplitConfig(BaseTransformConfig):
    name: Literal["split"] = "split"

    left: list[TransformConfig] = list_([])
    right: list[TransformConfig] = list_([])
    passthrough: bool | None = None


@config()
class SympyConfig(BaseTransformConfig):
    name: Literal["sympy"] = "sympy"
    expression: str = MISSING
    tolerance: float = 1e-4

    @post_processor
    def _validate_expression(self, cfg: MainConfig):
        from corerl.utils.sympy import is_expression, is_valid_expression, to_sympy

        if not is_expression(self.expression):
            raise ValueError(f"Invalid sympy expression format: {self.expression}")

        expr, _, _ = to_sympy(self.expression)
        if not is_valid_expression(expr):
            raise ValueError(f"Expression contains unsupported operations: {self.expression}")


TransformConfig = Annotated[
    AddRawConfig
    | BoundsConfig
    | DeltaConfig
    | IdentityConfig
    | NormalizerConfig
    | NukeConfig
    | SplitConfig
    | SympyConfig
    | TraceConfig,
    Field(discriminator="name"),
]
