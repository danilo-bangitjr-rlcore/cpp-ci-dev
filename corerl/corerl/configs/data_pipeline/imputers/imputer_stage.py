"""Per-tag imputer stage configuration."""

from typing import TYPE_CHECKING, Any, Literal

from lib_config.config import config
from pydantic import Field

from corerl.configs.data_pipeline.imputers.base import BaseImputerStageConfig
from corerl.configs.data_pipeline.imputers.per_tag.identity import IdentityImputerConfig

if TYPE_CHECKING:
    pass


@config()
class PerTagImputerConfig(BaseImputerStageConfig):
    name: Literal['per-tag'] = 'per-tag'
    default: Any = Field(default_factory=IdentityImputerConfig)
