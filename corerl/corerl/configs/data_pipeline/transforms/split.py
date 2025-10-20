"""Split transform configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lib_config.config import config, list_

from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig

if TYPE_CHECKING:
    pass


@config()
class SplitConfig(BaseTransformConfig):
    name: Literal['split'] = 'split'

    left: list[Any] = list_([])
    right: list[Any] = list_([])
    passthrough: bool | None = None
