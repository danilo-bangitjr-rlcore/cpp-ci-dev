
from typing import Literal

from lib_config.config import config

from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig


@config()
class AddRawConfig(BaseTransformConfig):
    name: Literal['add_raw'] = 'add_raw'
