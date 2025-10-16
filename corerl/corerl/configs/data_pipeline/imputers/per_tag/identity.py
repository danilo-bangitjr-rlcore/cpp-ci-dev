
from typing import Literal

from lib_config.config import config

from corerl.configs.data_pipeline.imputers.per_tag.base import BasePerTagImputerConfig


@config()
class IdentityImputerConfig(BasePerTagImputerConfig):
    name: Literal['identity'] = "identity"
