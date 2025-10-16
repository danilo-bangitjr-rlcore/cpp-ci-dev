
from typing import Literal

from lib_config.config import MISSING, config

from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig


@config()
class BoundsConfig(BaseTransformConfig):
    name: Literal['bounds'] = 'bounds'
    bounds: tuple[float, float] = MISSING
    mode: Literal['clip', 'nan'] = 'clip'
