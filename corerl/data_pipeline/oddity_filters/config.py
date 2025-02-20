from pydantic import Field

from corerl.configs.config import config
from corerl.data_pipeline.oddity_filters.factory import IdentityFilterConfig, OddityFilterConfig


@config()
class GlobalOddityFilterConfig:
    default: OddityFilterConfig = Field(default_factory=IdentityFilterConfig)
