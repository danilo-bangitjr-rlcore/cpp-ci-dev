
from typing import Literal

from lib_config.config import config

from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig


@config()
class NukeConfig(BaseTransformConfig):
    name: Literal['nuke'] = 'nuke'
