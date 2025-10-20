
from typing import Literal

from lib_config.config import config

from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig


@config()
class NormalizerConfig(BaseTransformConfig):
    name: Literal['normalize'] = 'normalize'
    min: float | None = None
    max: float | None = None
    from_data: bool = False
