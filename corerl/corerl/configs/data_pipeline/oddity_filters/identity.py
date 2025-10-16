
from typing import Literal

from lib_config.config import config

from corerl.configs.data_pipeline.oddity_filters.base import BaseOddityFilterConfig


@config()
class IdentityFilterConfig(BaseOddityFilterConfig):
    name: Literal['identity'] = 'identity'
