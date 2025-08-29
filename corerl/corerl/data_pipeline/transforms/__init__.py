from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from lib_config.config import MISSING, config, list_, post_processor
from pydantic import Field
from pydantic.dataclasses import rebuild_dataclass

from corerl.data_pipeline.transforms.add_raw import AddRawConfig
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.bounds import BoundsConfig
from corerl.data_pipeline.transforms.comparator import ComparatorConfig
from corerl.data_pipeline.transforms.delta import DeltaConfig
from corerl.data_pipeline.transforms.greater_than import GreaterThanConfig
from corerl.data_pipeline.transforms.identity import IdentityConfig
from corerl.data_pipeline.transforms.inverse import InverseConfig
from corerl.data_pipeline.transforms.less_than import LessThanConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.nuke import NukeConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.utils.sympy import is_expression, is_valid_expression, to_sympy

if TYPE_CHECKING:
    from corerl.config import MainConfig


"""
To avoid circular imports and partially defined types
that result in partially defined schemas, these configs
are defined in the same place as the union type TransformConfig.
"""
@config()
class SplitConfig(BaseTransformConfig):
    name: Literal['split'] = 'split'

    left: list[TransformConfig] = list_([IdentityConfig])
    right: list[TransformConfig] = list_([IdentityConfig])
    passthrough: bool | None = None


@config()
class SympyConfig(BaseTransformConfig):
    name: Literal["sympy"] = "sympy"
    expression: str = MISSING

    @post_processor
    def _validate_expression(self, cfg: MainConfig):
        # Validate sympy expression format and supported operations
        if not is_expression(self.expression):
            raise ValueError(f"Invalid sympy expression format: {self.expression}")

        expr, _, _ = to_sympy(self.expression)
        if not is_valid_expression(expr):
            raise ValueError(f"Expression contains unsupported operations: {self.expression}")


TransformConfig = Annotated[
    AddRawConfig
    | BoundsConfig
    | DeltaConfig
    | GreaterThanConfig
    | IdentityConfig
    | InverseConfig
    | LessThanConfig
    | NormalizerConfig
    | NukeConfig
    | SplitConfig
    | SympyConfig
    | TraceConfig
    | ComparatorConfig
, Field(discriminator='name')]


def register_dispatchers():
    from corerl.data_pipeline.transforms.split import SplitTransform
    from corerl.data_pipeline.transforms.sympy import SympyTransform

    transform_group.dispatcher(SplitTransform)
    transform_group.dispatcher(SympyTransform)

    # Because TransformConfig was only partially known when
    # pydantic first parsed these schemas, rebuild them
    # now that they are completely known.
    rebuild_dataclass(cast(Any, SplitConfig))
    rebuild_dataclass(cast(Any, SympyConfig))
