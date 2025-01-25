from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import Field
from pydantic.dataclasses import rebuild_dataclass
from typing_extensions import Annotated

from corerl.configs.config import MISSING, config, list_
from corerl.data_pipeline.transforms.add_raw import AddRawConfig
from corerl.data_pipeline.transforms.affine import AffineConfig
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.comparator import ComparatorConfig
from corerl.data_pipeline.transforms.delta import DeltaConfig
from corerl.data_pipeline.transforms.greater_than import GreaterThanConfig
from corerl.data_pipeline.transforms.identity import IdentityConfig
from corerl.data_pipeline.transforms.less_than import LessThanConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.null import NullConfig
from corerl.data_pipeline.transforms.power import PowerConfig
from corerl.data_pipeline.transforms.scale import ScaleConfig
from corerl.data_pipeline.transforms.trace import TraceConfig

"""
To avoid circular imports and partially defined types
that result in partially defined schemas, these two
configs are defined in the same place as the union
type TransformConfig.
"""
@config()
class BinaryConfig(BaseTransformConfig):
    name: Literal['binary'] = "binary"
    op: Literal['prod', 'min', 'max', 'add'] = MISSING

    other: str = MISSING
    other_xform: list[TransformConfig] = list_([IdentityConfig])



@config()
class SplitConfig(BaseTransformConfig):
    name: Literal['split'] = 'split'

    left: list[TransformConfig] = list_([IdentityConfig])
    right: list[TransformConfig] = list_([IdentityConfig])
    passthrough: bool | None = None


TransformConfig = Annotated[
    AddRawConfig
    | AffineConfig
    | DeltaConfig
    | GreaterThanConfig
    | IdentityConfig
    | LessThanConfig
    | NormalizerConfig
    | NullConfig
    | PowerConfig
    | BinaryConfig
    | ScaleConfig
    | SplitConfig
    | TraceConfig
    | ComparatorConfig
, Field(discriminator='name')]


def register_dispatchers():
    from corerl.data_pipeline.transforms.binary import BinaryTransform
    from corerl.data_pipeline.transforms.split import SplitTransform

    transform_group.dispatcher(BinaryTransform)
    transform_group.dispatcher(SplitTransform)

    # Because TransformConfig was only partially known when
    # pydantic first parsed these schemas, rebuild them
    # now that they are completely known.
    rebuild_dataclass(cast(Any, BinaryConfig))
    rebuild_dataclass(cast(Any, SplitConfig))
