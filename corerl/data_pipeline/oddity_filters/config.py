from corerl.configs.config import config, list_
from corerl.data_pipeline.oddity_filters.factory import IdentityFilterConfig, OddityFilterConfig


@config()
class GlobalOddityFilterConfig:
    defaults: list[OddityFilterConfig] = list_([IdentityFilterConfig()])
