
from typing import Any

from lib_config.config import MISSING, config


@config()
class BaseOddityFilterConfig:
    name: Any = MISSING
